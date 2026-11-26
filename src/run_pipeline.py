"""End-to-end time-aware fraud model comparison and evaluation."""
import argparse, gc, time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from .config import load_config
from .data_loader import load_data
from .diagnostics import fit_validation_calibrator,apply_calibrator,write_threshold_diagnostics,save_pr_overlay
from .evaluation import metrics,threshold_policies
from .explain import importance
from .features import engineer_features
from .imbalance import stratified_time_sample,undersample
from .models import MODEL_NAMES,build_model,parameter_candidates
from .splits import split_train_validation,time_split
from .visualization import setup

def _paths():
    for p in ('outputs/tables','outputs/figures','outputs/models','data/sample'):
        Path(p).mkdir(parents=True,exist_ok=True)

def _sample_fixture(seed):
    p=Path('data/sample/transactions_sample.csv')
    if not p.exists():
        from .data_generator import generate_transactions
        generate_transactions(5000,seed).to_csv(p,index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample',action='store_true'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    started=time.time(); cfg=load_config(); _paths(); _sample_fixture(args.seed)
    train_source,test_source,source=load_data(args.sample,args.seed)
    if test_source is None:
        # CI synthetic fixture follows a 60/20/20 time split.
        fit_raw,val_raw,test_raw=time_split(train_source)
    else:
        fit_raw,val_raw,test_raw=split_train_validation(train_source,test_source,cfg['validation_fraction'])
    fit_raw=fit_raw.copy(); val_raw=val_raw.copy(); test_raw=test_raw.copy()
    fit_raw['__split']='train'; val_raw['__split']='validation'; test_raw['__split']='test'
    combined=pd.concat([fit_raw,val_raw,test_raw],ignore_index=True)
    X,y,events=engineer_features(combined)
    partitions=events['__split'].to_numpy()
    fit_ix=np.flatnonzero(partitions=='train'); val_ix=np.flatnonzero(partitions=='validation'); test_ix=np.flatnonzero(partitions=='test')
    Xtr,ytr=X.iloc[fit_ix],y.iloc[fit_ix].to_numpy(); Xv,yv=X.iloc[val_ix],y.iloc[val_ix].to_numpy(); Xte,yte=X.iloc[test_ix],y.iloc[test_ix].to_numpy()
    tv=events.iloc[val_ix].trans_date_trans_time; tt=events.iloc[test_ix].trans_date_trans_time
    assert events.iloc[fit_ix].trans_date_trans_time.max()<tv.min()
    assert tv.max()<tt.min()
    # Small randomized parameter search uses a chronological/class-stratified training subsample.
    Xsearch,ysearch=stratified_time_sample(Xtr,pd.Series(ytr),events.iloc[fit_ix].trans_date_trans_time,cfg['search_train_rows'],args.seed)
    scale=max(1.,float((ysearch==0).sum()/max(1,(ysearch==1).sum())))
    params_by_model={}; tuning_rows=[]
    for name in MODEL_NAMES:
        candidates=parameter_candidates(args.seed)[name][:cfg['search_candidates']]
        strat='none' if name=='Neural Network' else 'class_weight'
        best_score=-1.; best_params=candidates[0]
        for params in candidates:
            model=build_model(name,args.seed,scale,strat,params); model.fit(Xsearch,ysearch)
            score=average_precision_score(yv,model.predict_proba(Xv)[:,1]) if yv.sum() else 0.
            tuning_rows.append({'model':name,'parameters':str(params),'validation_pr_auc':score})
            if score>best_score: best_score,best_params=score,params
            del model; gc.collect()
        params_by_model[name]=best_params
    pd.DataFrame(tuning_rows).to_csv('outputs/tables/tuning_results.csv',index=False)
    # Fit every applicable imbalance option on the same time-stratified fit sample.
    Xfit,yfit=stratified_time_sample(Xtr,pd.Series(ytr),events.iloc[fit_ix].trans_date_trans_time,cfg['fit_train_rows'],args.seed+1)
    Xsm,ysm=stratified_time_sample(Xtr,pd.Series(ytr),events.iloc[fit_ix].trans_date_trans_time,cfg['smote_train_rows'],args.seed+2)
    strategy_rows=[]; winning_models={}; val_scores={}; strategy_best={}
    for name in MODEL_NAMES:
        strategies=['none','class_weight','undersampling','smote'] if name!='Neural Network' else ['none','undersampling','smote']
        best_ap=-1.; best_model=None; best_strategy=None
        for si,strategy in enumerate(strategies):
            if strategy=='undersampling': Xa,ya=undersample(Xfit,yfit,args.seed+si,cfg['undersample_max_ratio'])
            elif strategy=='smote':
                sampler=SMOTE(random_state=args.seed+si,k_neighbors=min(5,max(1,int((ysm==1).sum()-1))))
                Xa,ya=sampler.fit_resample(Xsm,ysm)
            else: Xa,ya=Xfit,yfit
            model=build_model(name,args.seed+si,scale,strategy,params_by_model[name]); model.fit(Xa,ya)
            pv=model.predict_proba(Xv)[:,1]; ap=average_precision_score(yv,pv) if yv.sum() else 0.
            strategy_rows.append({'model':name,'strategy':strategy,'validation_pr_auc':ap,'train_rows_after_sampling':len(Xa),
                                  'parameters':str(params_by_model[name])})
            if ap>best_ap: best_ap,best_model,best_strategy=ap,model,strategy
            else: del model
            gc.collect()
        winning_models[name]=best_model; val_scores[name]=best_ap; strategy_best[name]=best_strategy
        joblib.dump(best_model,f"outputs/models/{name.lower().replace(' ','_')}.joblib")
    strategy_frame=pd.DataFrame(strategy_rows); strategy_frame['selected_for_model']=strategy_frame.apply(lambda r:r.strategy==strategy_best[r.model],axis=1)
    strategy_frame.to_csv('outputs/tables/imbalance_comparison.csv',index=False)
    # Calibrate on validation's first chronological half; thresholds use only its second half.
    val_cut=len(yv)//2; y_policy=yv[val_cut:]
    modelscores={}; selected_policies={}; calibration_rows=[]; all_policy_rows=[]
    capacity_rates=tuple(cfg['review_capacity_rates'])
    for name,model in winning_models.items():
        raw_v=model.predict_proba(Xv)[:,1]; raw_t=model.predict_proba(Xte)[:,1]
        calibrator,cal_name,cal_loss=fit_validation_calibrator(yv,raw_v)
        cal_v=apply_calibrator(calibrator,cal_name,raw_v); cal_t=apply_calibrator(calibrator,cal_name,raw_t)
        modelscores[name]={'uncalibrated_validation':raw_v,'uncalibrated_test':raw_t,'calibrated_validation':cal_v,'calibrated_test':cal_t}
        calibration_rows.append({'model':name,'calibrator':cal_name,'second_half_validation_log_loss':cal_loss,
                                 'validation_pr_auc_uncalibrated':average_precision_score(yv,raw_v),
                                 'validation_pr_auc_calibrated':average_precision_score(yv,cal_v)})
        selected_policies[name]={}
        for mode,sv,st in [('uncalibrated',raw_v,raw_t),('calibrated',cal_v,cal_t)]:
            policies,sweep=threshold_policies(y_policy,sv[val_cut:],cfg['precision_target'],cfg['minimum_validation_true_positives'],capacity_rates)
            selected_policies[name][mode]=policies
            if name==max(val_scores,key=val_scores.get) and mode=='calibrated':
                pd.DataFrame(sweep).to_csv('outputs/tables/thresholds.csv',index=False)
            for policy,point in policies.items():
                m=metrics(yte,st,point['threshold'])
                all_policy_rows.append({'model':name,'scores':mode,'policy':policy,'threshold':point['threshold'],
                    'validation_precision':point.get('validation_precision'),'validation_recall':point.get('validation_recall'),
                    'validation_true_positives':point.get('validation_tp'),'policy_note':point.get('note'),**m})
    pd.DataFrame(calibration_rows).to_csv('outputs/tables/calibration_comparison.csv',index=False)
    policy_frame=pd.DataFrame(all_policy_rows); policy_frame.to_csv('outputs/tables/policy_comparison.csv',index=False)
    final_model=max(val_scores,key=val_scores.get); final_policy='f1_max'; final_scores=modelscores[final_model]['calibrated_test']
    final_threshold=selected_policies[final_model]['calibrated'][final_policy]['threshold']
    final_metrics=metrics(yte,final_scores,final_threshold)
    # Legacy compact model table now reflects the chosen calibrated F1 policy.
    compact=policy_frame[(policy_frame.scores=='calibrated')&(policy_frame.policy=='f1_max')].copy()
    compact.to_csv('outputs/tables/model_comparison.csv',index=False)
    rng=np.random.default_rng(args.seed); boots=[]
    for _ in range(cfg['bootstrap_resamples']):
        ix=rng.integers(0,len(yte),len(yte)); mm=metrics(yte[ix],final_scores[ix],final_threshold); boots.append([mm['precision'],mm['recall']])
    ci=np.percentile(boots,[2.5,97.5],axis=0)
    pd.DataFrame([{'metric':'precision','lower_95':ci[0,0],'upper_95':ci[1,0]},{'metric':'recall','lower_95':ci[0,1],'upper_95':ci[1,1]}]).to_csv('outputs/tables/bootstrap_ci.csv',index=False)
    # Slice analysis on the one fixed final policy.
    final_test=events.iloc[test_ix].copy(); final_test['score']=final_scores; final_test['amount_bucket']=pd.qcut(final_test.amt,4,duplicates='drop')
    slice_rows=[]
    for col in ('amount_bucket','category'):
        for key,rows in final_test.groupby(col,observed=True):
            slice_rows.append({'slice':col,'value':str(key),**metrics(rows.is_fraud,rows.score,final_threshold)})
    pd.DataFrame(slice_rows).to_csv('outputs/tables/slice_analysis.csv',index=False)
    # Test-set graphics and diagnosis. No threshold is selected from test results.
    setup(); plt.figure(); plt.bar(['legitimate','fraud'],[(y==0).sum(),(y==1).sum()]); plt.title('Class imbalance across timeline'); plt.ylabel('Transactions'); plt.tight_layout(); plt.savefig('outputs/figures/class_imbalance.png',dpi=150); plt.close()
    fig,ax=plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(yte,final_scores>=final_threshold),display_labels=['legitimate','fraud']).plot(ax=ax,colorbar=False); ax.set_title(f'{final_model}: calibrated test F1 policy'); fig.tight_layout(); fig.savefig('outputs/figures/confusion_matrix.png',dpi=150); plt.close(fig)
    val_raw=modelscores[final_model]['calibrated_validation'][val_cut:]; y_for_threshold=y_policy
    thpol,throws=threshold_policies(y_for_threshold,val_raw,cfg['precision_target'],cfg['minimum_validation_true_positives'],capacity_rates)
    pd.DataFrame(throws).to_csv('outputs/tables/threshold_sweep.csv',index=False)
    plt.figure(figsize=(7,4)); sf=pd.DataFrame(throws)
    plt.plot(sf.threshold,sf.precision,label='Precision'); plt.plot(sf.threshold,sf.recall,label='Recall'); plt.plot(sf.threshold,sf.f1 if 'f1' in sf else 2*sf.precision*sf.recall/(sf.precision+sf.recall+1e-12),label='F1')
    plt.xlabel('Validation threshold'); plt.ylabel('Metric'); plt.title('Validation threshold sweep'); plt.legend(); plt.tight_layout(); plt.savefig('outputs/figures/threshold_sweep.png',dpi=150); plt.close()
    cost_rows=[]; amount=final_test.amt.to_numpy()
    for threshold in np.unique(np.r_[0,np.quantile(final_scores,np.linspace(0,1,301)),1,final_threshold]):
        alerted=final_scores>=threshold; false_alerts=alerted&(yte==0)
        cost=float(amount[(yte==1)&(~alerted)].sum()+cfg['review_cost']*false_alerts.sum()); cost_rows.append({'threshold':threshold,'total_cost':cost})
    pd.DataFrame(cost_rows).to_csv('outputs/tables/cost_curve.csv',index=False)
    final_alert=final_scores>=final_threshold
    missed_amount=float(amount[(yte==1)&(~final_alert)].sum())
    review_amount=float(cfg['review_cost']*(final_alert&(yte==0)).sum())
    pd.DataFrame([{'final_threshold':final_threshold,'alert_count':int(final_alert.sum()),'missed_fraud_amount':missed_amount,
      'false_alarm_review_cost':review_amount,'total_cost':missed_amount+review_amount,
      'all_missed_cost':float(amount[yte==1].sum()),'review_every_transaction_cost':float(cfg['review_cost']*(yte==0).sum())}]).to_csv('outputs/tables/cost_summary.csv',index=False)
    plt.figure(); plt.plot([r['threshold'] for r in cost_rows],[r['total_cost'] for r in cost_rows]); plt.xlabel('Threshold'); plt.ylabel('Estimated cost'); plt.title('Test cost versus threshold (diagnostic only)'); plt.tight_layout(); plt.savefig('outputs/figures/cost_curve.png',dpi=150); plt.close()
    strategy_frame.groupby('strategy').validation_pr_auc.mean().plot.bar(); plt.ylabel('Mean validation PR-AUC'); plt.title('Imbalance strategy comparison'); plt.tight_layout(); plt.savefig('outputs/figures/imbalance_comparison.png',dpi=150); plt.close()
    Xperm=Xv.iloc[val_cut:]; yperm=y_policy
    ixp=np.linspace(0,len(Xperm)-1,min(cfg['permutation_rows'],len(Xperm)),dtype=int)
    importance_frame=importance(winning_models[final_model],Xperm.iloc[ixp],yperm[ixp],args.seed)
    importance_frame.to_csv('outputs/tables/feature_importance.csv',index=False)
    importance_frame.head(20).sort_values('importance').plot.barh(x='feature',y='importance',legend=False,figsize=(8,6))
    plt.title('Validation permutation importance'); plt.tight_layout(); plt.savefig('outputs/figures/feature_importance.png',dpi=150); plt.close()
    write_threshold_diagnostics('outputs/tables/threshold_diagnostics.csv','outputs/figures/score_distributions.png',modelscores,yv,yte,selected_policies,final_model)
    save_pr_overlay('outputs/figures/validation_test_pr_curves.png',modelscores,yv,yte)
    pd.DataFrame([{'data_source':source,'rows_total':len(combined),'source_train_rows':len(train_source),'source_test_rows':len(test_source) if test_source is not None else 0,
      'train_rows':len(fit_ix),'validation_rows':len(val_ix),'test_rows':len(test_ix),'train_fraud_rate':float(ytr.mean()),'validation_fraud_rate':float(yv.mean()),'test_fraud_rate':float(yte.mean()),
      'best_model':final_model,'best_imbalance_strategy':strategy_best[final_model],'final_policy':final_policy,'chosen_threshold':final_threshold,
      'test_precision':final_metrics['precision'],'test_recall':final_metrics['recall'],'test_f1':final_metrics['f1'],'test_pr_auc':final_metrics['pr_auc'],
      'test_tn':final_metrics['tn'],'test_fp':final_metrics['fp'],'test_fn':final_metrics['fn'],'test_tp':final_metrics['tp'],
      'runtime_seconds':time.time()-started}]).to_csv('outputs/tables/run_summary.csv',index=False)
    print(pd.read_csv('outputs/tables/run_summary.csv').to_string(index=False))

if __name__=='__main__': main()
