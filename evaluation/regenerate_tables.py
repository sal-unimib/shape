"""Rebuild thesis evaluation tables from retained SHAPE outputs, without inference.

Run from any directory: python evaluation/regenerate_tables.py
Only evaluation CSV/JSON/LaTeX exports are written. Source data, output/ and logs/
are read-only. RQ3 uses the thesis Exact/Different/Missing comparison criteria;
its export replaces the legacy aggregate TP/FP/FN schema.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DETAILS = ROOT / 'evaluation' / 'details'
DETAILS.mkdir(exist_ok=True)
import hashlib
INPUT_PATHS = sorted([*ROOT.glob('dataset/**/*.json'), *ROOT.glob('output/**/*.json'), *ROOT.glob('logs/*.log')])
INPUT_HASHES = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUT_PATHS}


import json,csv
from pathlib import Path
from collections import Counter,defaultdict
r=ROOT
models=['mistral','mixtral_8x7','mixtral_8x22','llama','phi4']
elements=json.loads((r/'dataset/groundtruth/all_elements.json').read_text())
matches=defaultdict(dict)
with (r/'evaluation/rq1/helpers/actions_matching.csv').open(encoding='utf-8-sig') as f:
 for row in csv.DictReader(f,delimiter=';'):
  if row['gt_action_id']:matches[row['llm'],row['plan_id']][row['gt_action_id']]=row['gen_action_id'] or None

def decode(v):
 try:return json.loads(v)
 except (TypeError,ValueError):return v

def codings(v):
 if isinstance(v,dict):
  if isinstance(v.get('coding'),list):return {(str(c.get('system','')),str(c.get('code',''))) for c in v['coding'] if c.get('code') is not None}
  if 'code' in v and 'system' in v:return {(str(v['system']),str(v['code']))}
 return set()
summaries={};details=[]
for model in models:
 c=Counter();cats=defaultdict(Counter);bad_examples=[]
 for pn in elements:
  gt=json.loads((r/f'dataset/groundtruth/{pn}.json').read_text())
  gen=json.loads((r/f'output/{model}/{pn}_plandefinition.json').read_text())
  gen_actions={a['linkId']:a for a in gen.get('action',[])}
  inter=json.loads((r/f'output/{model}/{pn}_output.json').read_text())
  inters={a['id']:a for a in inter.get('fhir_actions',[])}
  anns={a['id']:a for a in elements[pn]['actions']}
  for ga in gt['action']:
   gid=matches[model,pn].get(ga['linkId']);a=gen_actions.get(gid)
   ann=anns[ga['linkId']];pars={p['path']:p for p in ann.get('parameters') or []}
   for field in ['timingTiming','offsetDuration']:
    tv=ga.get(field) if field=='timingTiming' else next(iter(ga.get('relatedAction') or [{}]),{}).get(field)
    if tv is not None:
     av=(a.get(field) if field=='timingTiming' else next(iter(a.get('relatedAction') or [{}]),{}).get(field)) if a else None
     c['temporal_'+('missing' if av is None else 'exact' if av==tv else 'different')]+=1
   for dv in ga.get('dynamicValue') or []:
    path=dv['path'];annp=pars.get(path,{})
    cat=annp.get('category','unknown');gtv=decode(dv['expression'].get('expression'))
    gendv={v['path']:v for v in (a.get('dynamicValue') or [])} if a else {}
    v=gendv.get(path);gv=decode(v['expression'].get('expression')) if v else None
    old='missing' if not v else 'exact' if gv==gtv else 'different'
    vals=[annp['value']] if annp.get('value') is not None else annp.get('possible_values',[])
    gt_codes=codings(gtv);gen_codes=codings(gv)
    accepted=set().union(*(codings(x) for x in vals)) if vals else gt_codes
    value_ok=(bool(gen_codes & accepted) if cat=='ontology' and accepted else (gv.get('code') if isinstance(gv,dict) else gv)==gtv if cat=='value_set' else gv==gtv)
    status=('missing_action' if a is None else 'missing_path') if not v else 'correct' if value_ok else 'different'
    direct_code='missing' if not v else 'correct' if (bool(gen_codes & gt_codes) if gt_codes else gv==gtv) else 'different'
    c['param_'+old]+=1;c['revised_'+status]+=1;cats[cat][status]+=1
    c['single_ref_'+direct_code]+=1
    if old=='different' and gt_codes and gen_codes & gt_codes:c['text_only_coded_difference']+=1
    if v and annp.get('category') in ['ontology','value_set']:
     c['coded_expected_present']+=1
     if value_ok:c['coded_correct']+=1
    if a:
     ia=inters.get(gid,{})
     if ia.get('activity_definition'):
      expected=ann['activity_definition'].split('/')[-1]
      if str(ia['activity_definition']['id'])!=expected:c['param_on_wrong_ad']+=1
    details.append(dict(model=model,plan=pn,gt_action=ga['linkId'],gen_action=gid,path=path,category=cat,old=old,status=status,single_reference=direct_code,gt_value=gtv,gen_value=gv,accepted_values=vals))
  # extra paths on matched actions (excluded by notebook)
  for ga in gt['action']:
   gid=matches[model,pn].get(ga['linkId']);a=gen_actions.get(gid)
   if a:
    exp={dv['path'] for dv in ga.get('dynamicValue') or []}
    c['extra_paths_matched_actions']+=sum(d['path'] not in exp for d in a.get('dynamicValue') or [])
 summaries[model]={'counts':dict(c),'categories':{k:dict(v) for k,v in cats.items()}}

(DETAILS/'rq3_encoding_summary.json').write_text(json.dumps(summaries,indent=2))
(DETAILS/'rq3_parameter_details.json').write_text(json.dumps(details,indent=2))

import json,csv,re,ast
from pathlib import Path
from collections import Counter,defaultdict
r=ROOT;out=DETAILS
models=['mistral','mixtral_8x7','mixtral_8x22','llama','phi4']
elements=json.load(open(r/'dataset/groundtruth/all_elements.json'));matches=defaultdict(dict)
for row in csv.DictReader(open(r/'evaluation/rq1/helpers/actions_matching.csv',encoding='utf-8-sig'),delimiter=';'):
 if row['gt_action_id']:matches[row['llm'],row['plan_id']][row['gt_action_id']]=row['gen_action_id'] or None

def csvout(name,rows):
 if not rows:return
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows])
# MI12: derive all per-action outcomes from saved artifacts.
ads={}
for p in (r/'dataset/activitydefinition').glob('*.json'):
 a=json.load(open(p));ads[a['id']]=a['title']
adrows=[]
for m in models:
 for pn,ann in elements.items():
  inter=json.load(open(r/f'output/{m}/{pn}_output.json'));acts={a['id']:a for a in inter['fhir_actions']}
  for gt in ann['actions']:
   genid=matches[m,pn].get(gt['id']);a=acts.get(genid)
   if not a:continue
   exp=gt['activity_definition'].split('/')[-1];cs=a['candidates'];rank=next((c['retrieval_rank'] for c in cs if str(c['activity_definition']['id'])==exp),None)
   selected=str(a['activity_definition']['id']) if a.get('activity_definition') else None
   outcome='retrieval_error' if rank is None else 'correct' if selected==exp else 'selection_error'
   adrows.append(dict(model=m,plan=pn,reference_action=gt['id'],generated_action=genid,reference_label=gt['label'],generated_label=a['label'],expected_id=exp,expected_title=ads.get(exp),candidates=[{'rank':c['retrieval_rank'],'id':c['activity_definition']['id'],'title':c['activity_definition']['title']} for c in cs],reference_rank=rank,selected_id=selected,selected_title=ads.get(selected),outcome=outcome))
csvout('MI12_activity_grounding_cases.csv',adrows)
adsummary=[]
for m in models:
 rows=[x for x in adrows if x['model']==m];c=Counter(x['outcome'] for x in rows)
 adsummary.append(dict(model=m,N=len(rows),retrieval_errors=c['retrieval_error'],selection_errors=c['selection_error'],correct=c['correct']))
csvout('MI12_summary.csv',adsummary)
byad=[]
for aid in sorted(ads):
 rows=[x for x in adrows if x['expected_id']==aid];c=Counter(x['outcome'] for x in rows)
 byad.append(dict(expected_id=aid,title=ads[aid],N=len(rows),retrieval_errors=c['retrieval_error'],selection_errors=c['selection_error'],correct=c['correct']))
csvout('MI12_by_activity.csv',byad)
# MI07: logs contain two complete runs. Match warning payloads to retained intermediate values.
warnings=[]
for log in sorted((r/'logs').glob('*.log')):
 lines=log.read_text().splitlines();model=None;plan=None
 for i,line in enumerate(lines):
  start=re.search(r'\| shape \| (\S+) - (P\d+) - SHAPE Pipeline started',line)
  if start:model,plan=start.groups()
  msg=re.search(r'\| shape.stages.stage5 \| \S+ - (P\d+) - (A\d+) - Error creating (Timing|Duration) from FHIRTemporalConstraint:\s*(.*)',line)
  if msg:
   p,a,kind,payload=msg.groups()
   if not payload and i+1<len(lines):payload=lines[i+1]
   try:value=ast.literal_eval(payload)
   except Exception:value=payload
   warnings.append(dict(model=model,plan=p,action=a,kind=kind,value=value,log=log.name,line=i+1))
temporal=[]
for m in models:
 for pn in elements:
  inter=json.load(open(r/f'output/{m}/{pn}_output.json'));final=json.load(open(r/f'output/{m}/{pn}_plandefinition.json'));fa={a['linkId']:a for a in final['action']}
  for a in inter['fhir_actions']:
   tc=a.get('fhir_temporal_constraint')
   if not tc:continue
   typ=tc['type'];dest=fa[a['id']]
   if typ in ['DURATION','REPEAT']:present='timingTiming' in dest;kind='Timing';has_target=True
   else:present=any('offsetDuration' in rel for rel in dest.get('relatedAction',[]));kind='Duration';has_target=bool(dest.get('relatedAction'))
   evidence=[w for w in warnings if w['model']==m and w['plan']==pn and w['action']==a['id'] and w['kind']==kind and w['value']==tc['value']]
   state='transferred' if present else 'validation_omission_log_supported' if evidence else 'no_incoming_relationship' if typ=='OFFSET' and not has_target else 'omitted_cause_not_attributable'
   temporal.append(dict(model=m,plan=pn,generated_action=a['id'],type=typ,state=state,intermediate_value=tc['value'],matching_warnings=[{'log':w['log'],'line':w['line']} for w in evidence]))
csvout('MI07_intermediate_temporal_cases.csv',temporal)
# Reconcile the 26 expected constraints, distinguishing missing earlier from construction loss.
expected=[]
for m in models:
 for pn in elements:
  gt=json.load(open(r/f'dataset/groundtruth/{pn}.json'));gen=json.load(open(r/f'output/{m}/{pn}_plandefinition.json'));ga={a['linkId']:a for a in gen['action']}
  inter={a['id']:a for a in json.load(open(r/f'output/{m}/{pn}_output.json'))['fhir_actions']}
  for a in gt['action']:
   vals=[]
   if a.get('timingTiming') is not None:vals.append(('Timing',a['timingTiming']))
   for rel in a.get('relatedAction',[]):
    if rel.get('offsetDuration') is not None:vals.append(('Duration',rel['offsetDuration']))
   for kind,v in vals:
    gid=matches[m,pn].get(a['linkId']);g=ga.get(gid)
    got=g.get('timingTiming') if kind=='Timing' and g else next((rel['offsetDuration'] for rel in g.get('relatedAction',[]) if 'offsetDuration' in rel),None) if g else None
    status='missing' if got is None else 'exact' if got==v else 'different'
    it=inter.get(gid,{})
    case=next((x for x in temporal if x['model']==m and x['plan']==pn and x['generated_action']==gid),None)
    cause=None
    if status=='missing':cause='missing_action' if g is None else case['state'] if case else 'no_encoded_temporal_constraint'
    expected.append(dict(model=m,plan=pn,reference_action=a['linkId'],generated_action=gid,kind=kind,status=status,missing_cause=cause))
csvout('MI07_expected_temporal_cases.csv',expected)
ts=[]
for m in models:
 rows=[x for x in temporal if x['model']==m];c=Counter(x['state'] for x in rows);ex=[x for x in expected if x['model']==m];e=Counter(x['status'] for x in ex);ca=Counter(x['missing_cause'] for x in ex if x['status']=='missing')
 ts.append(dict(model=m,intermediate_constraints=len(rows),transferred=c['transferred'],log_supported_omissions=c['validation_omission_log_supported'],affected_pathways=len({x['plan'] for x in rows if x['state']=='validation_omission_log_supported'}),other_omissions=len(rows)-c['transferred']-c['validation_omission_log_supported'],expected_exact=e['exact'],expected_different=e['different'],expected_missing=e['missing'],expected_missing_causes=dict(ca)))
csvout('MI07_summary.csv',ts)
# MI15: use the corrected coded-value comparison already used in the thesis audit.
details=json.load(open(DETAILS/'rq3_parameter_details.json'));llama=[x for x in details if x['model']=='llama'];diff=[x for x in llama if x['status']=='different']
for x in diff:
 x['disagreement_class']='ontology_unresolved_value' if x['category']=='ontology' and x['gen_value'] in [None,'None'] else 'ontology_different_code' if x['category']=='ontology' else 'reference_text_difference' if x['category']=='reference' else 'string_value_difference'
csvout('MI15_llama_26_differences.csv',diff)
csvout('MI15_llama_all_77.csv',llama)
# MI14: export all 90 configurations with explicit denominators.
rows=list(csv.DictReader(open(r/'evaluation/rq2/ontology_grounding_strategies/all_onto_strategy_analysis.csv')));assert len(rows)==90
onto=[]
for row in rows:
 k=int(row['candidates'][1:]);N=int(row['n_params']);H=int(row[f'hits_k{k}']);C=int(row['correct']);assert abs(float(row['selection_accuracy'])-(C/H if H else 0))<1e-10
 onto.append(dict(model=row['llm'],embedding=row['embedding'],strategy=row['strategy'],k=k,N=N,H=H,C=C,recall_at_1=float(row['recall_at_1']),recall_at_k=H/N,selection_accuracy=C/H if H else None,end_to_end_accuracy=C/N,retrieval_errors=int(row['retrieval_errors']),wrong_selections=int(row['selection_errors']),rejections_with_reference_available=H-C-int(row['selection_errors'])))
csvout('MI14_ontology_grounding_90_configurations.csv',onto)
# LaTeX appendix-ready longtable; no thesis source modifications.
tex=[r'\section{Ontology Grounding Results by Language Model}',r'\label{app:shape-ontology-by-model}',r'Each row reports eligible parameters ($N$), retrieval hits ($H_k$), and correct selections ($C$). Retrieval recall is $H_k/N$, conditional selection accuracy is $C/H_k$, and end-to-end accuracy is $C/N$. The evaluated subset is conditional on the preceding stages. Strategy labels denote term isolation, term mean pooling, and term concatenation.',r'\begingroup',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{lllrrrrrr}',r'\caption{SHAPE/RQ2 ontology grounding by configuration and language model. Counts accompany conditional and end-to-end accuracy.}\label{tab:shape-ontology-detailed}\\',r'\toprule',r'Embedding & Strategy & LLM & $k$ & $H_k/N$ & $C/H_k$ & $C/N$ & Sel. acc. & E2E acc.\\',r'\midrule',r'\endfirsthead',r'\toprule',r'Embedding & Strategy & LLM & $k$ & $H_k/N$ & $C/H_k$ & $C/N$ & Sel. acc. & E2E acc.\\',r'\midrule',r'\endhead']
for row in sorted(onto,key=lambda x:(x['embedding'],x['strategy'],x['k'],models.index(x['model']))):
 label={'mistral':'Mistral','mixtral_8x7':'Mixtral 8x7B','mixtral_8x22':'Mixtral 8x22B','llama':'Llama','phi4':'Phi-4'}[row['model']]
 strategy={'term':'Isolation','avg':'Mean pooling','full':'Concatenation'}[row['strategy']]
 tex.append(f"{row['embedding']} & {strategy} & {label} & {row['k']} & {row['H']}/{row['N']} & {row['C']}/{row['H']} & {row['C']}/{row['N']} & {row['selection_accuracy']:.3f} & {row['end_to_end_accuracy']:.3f} \\\\")
tex += [r'\bottomrule',r'\end{longtable}',r'\endgroup']
(out/'MI14_appendix_table.tex').write_text('\n'.join(tex)+'\n')
summary={'MI07':ts,'MI12':adsummary,'MI12_by_activity':byad,'MI15_classes':dict(Counter(x['disagreement_class'] for x in diff)),'MI14_N':{m:sorted({x['N'] for x in onto if x['model']==m}) for m in models},'MI14_selection_one':[{k:x[k] for k in ['model','embedding','strategy','k','H','C','N']} for x in onto if x['selection_accuracy']==1]}
(out/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));

allrows=[];mismatches=[]
aggregate=list(csv.DictReader(open(out/'MI14_ontology_grounding_90_configurations.csv')))
for embedding in ['nomic','sapbert']:
 candidates=json.load(open(r/f'evaluation/rq2/ontology_grounding_strategies/{embedding}_candidates.json'))
 resolutions=json.load(open(r/f'evaluation/rq2/ontology_grounding_strategies/{embedding}_parameter_resolution.json'))
 for model in models:
  for plan,ann in elements.items():
   actions={a['id']:a for a in json.load(open(r/f'output/{model}/{plan}_output.json'))['fhir_actions']}
   for gt in ann['actions']:
    gid=matches[model,plan].get(gt['id']);a=actions.get(gid)
    if not a or not a.get('activity_definition') or str(a['activity_definition']['id'])!=gt['activity_definition'].split('/')[-1]:continue
    identified={p['path'] for p in a['parameters']}
    for par in gt.get('parameters') or []:
     path=par['path']
     if par['category']!='ontology' or path not in identified:continue
     allowed=[par['value']] if par.get('value') else par.get('possible_values',[])
     codes={c['code'] for v in allowed for c in v.get('coding',[])}
     for level in ['term','avg','full']:
      cs=candidates[model][plan][gid][path][level]
      for k in [10,20,50]:
       hits=[c['retrieval_rank'] for c in cs if c['retrieval_rank']<=k and c['ontology_concept']['concept_id'] in codes]
       choice=resolutions[model][plan][gid][path][level]['k'+str(k)]['selected_concept']
       selected=choice.get('id') if choice else None
       allrows.append(dict(model=model,embedding=embedding,strategy=level,k=k,plan=plan,reference_action=gt['id'],generated_action=gid,path=path,allowed_codes=sorted(codes),reference_rank=min(hits) if hits else None,selected_code=selected,hit=bool(hits),correct=bool(hits) and selected in codes,unique_candidate_concepts=len({c['ontology_concept']['concept_id'] for c in cs if c['retrieval_rank']<=k})))
for row in aggregate:
 subset=[x for x in allrows if all(str(x[k])==row[k] for k in ['model','embedding','strategy','k'])]
 computed=(len(subset),sum(x['hit'] for x in subset),sum(x['correct'] for x in subset))
 reported=tuple(int(row[k]) for k in ['N','H','C'])
 if computed!=reported:mismatches.append((row,computed))
csvout('MI14_ontology_parameter_cases.csv',allrows)
print('Raw cases',len(allrows),'configuration mismatches',len(mismatches))
assert not mismatches, 'Stored ontology aggregates disagree with raw cases'



# Conventional repository exports, preserving RQ2 column names.
def export_at(relative, records):
    target=ROOT/relative
    with target.open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)

ad_export=[];confidence_export=[];param_export=[]
for model in models:
    rows=[x for x in adrows if x['model']==model]
    N=len(rows);hits={k:sum(x['reference_rank'] is not None and x['reference_rank']<=k for x in rows) for k in (1,2,3)}
    C=sum(x['outcome']=='correct' for x in rows)
    no_selection=sum(x['selected_id'] is None for x in rows)
    ad_export.append(dict(llm=model,recall_at_1=hits[1]/N,recall_at_2=hits[2]/N,recall_at_3=hits[3]/N,
        mrr=sum(1/x['reference_rank'] for x in rows if x['reference_rank'] is not None)/N,
        selection_accuracy=C/hits[3] if hits[3] else 0,e2e_accuracy=C/N,
        retrieval_error_rate=sum(x['outcome']=='retrieval_error' for x in rows)/N,
        selection_error_rate=sum(x['outcome']=='selection_error' for x in rows)/N,
        n_actions=N,hits_k1=hits[1],hits_k2=hits[2],hits_k3=hits[3],correct=C,
        retrieval_errors=sum(x['outcome']=='retrieval_error' for x in rows),
        selection_errors=sum(x['outcome']=='selection_error' for x in rows),none_selection=no_selection))
    cc=Counter()
    for x in rows:
        a=next(a for a in json.load(open(ROOT/f"output/{model}/{x['plan']}_output.json"))['fhir_actions'] if a['id']==x['generated_action'])
        confidence=(a.get('confidence') or 'NONE').lower()
        correct=x['outcome']=='correct';available=x['reference_rank'] is not None
        if confidence=='none':cc['none_gt_in_cand' if available else 'none_gt_not_in_cand']+=1
        else:
            cc[('correct_' if correct else 'not_correct_')+confidence]+=1
            if not correct and not available:cc['not_correct_'+confidence+'_no_cand']+=1
    fields=['correct_high','correct_medium','correct_low','not_correct_high','not_correct_medium','not_correct_low',
            'not_correct_high_no_cand','not_correct_medium_no_cand','not_correct_low_no_cand','none_gt_in_cand','none_gt_not_in_cand']
    confidence_export.append(dict(llm=model,**{key:cc[key] for key in fields}))
    assert sum(cc[key] for key in fields if not key.endswith('_no_cand'))==N
    # Follow the existing parameter-identification convention, including the
    # inverse mapping for FN on unique generated actions.
    tp=fp=fn=0
    for plan,annotation in elements.items():
        acts={a['id']:a for a in json.load(open(ROOT/f'output/{model}/{plan}_output.json'))['fhir_actions']}
        refs={a['id']:a for a in annotation['actions']}
        inverse={gid:gtid for gtid,gid in matches[model,plan].items() if gid}
        correct_refs={x['reference_action']:x['generated_action'] for x in rows if x['plan']==plan and x['outcome']=='correct'}
        for gtid,gid in correct_refs.items():
            wanted={p['path'] for p in refs[gtid].get('parameters') or []}
            generated={p['path'] for p in acts[gid].get('parameters') or []}
            tp+=len(wanted & generated);fp+=len(generated-wanted)
        for gid in set(correct_refs.values()):
            gtid=inverse[gid];wanted={p['path'] for p in refs[gtid].get('parameters') or []}
            generated={p['path'] for p in acts[gid].get('parameters') or []}
            fn+=len(wanted-generated)
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    param_export.append({'llm':model,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
                         'f1-score':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
export_at('evaluation/rq2/ad_grounding_evaluation_summary.csv',ad_export)
export_at('evaluation/rq2/ad_confidence_analysis.csv',confidence_export)
export_at('evaluation/rq2/param_identification_analysis.csv',param_export)
encoding=[]
for model in models:
    count=summaries[model]['counts']
    row={'llm':model,**{key:count.get(key,0) for key in ('temporal_exact','temporal_different','temporal_missing')}}
    row.update(parameter_exact=count.get('revised_correct',0),parameter_different=count.get('revised_different',0),
               parameter_missing=count.get('revised_missing_action',0)+count.get('revised_missing_path',0))
    assert sum(row[k] for k in ('temporal_exact','temporal_different','temporal_missing'))==26
    assert sum(row[k] for k in ('parameter_exact','parameter_different','parameter_missing'))==77
    encoding.append(row)
export_at('evaluation/rq3/e2e_analysis.csv',encoding)
# The complete ontology aggregates are re-exported from verified raw cases.
ontology_original=list(csv.DictReader(open(ROOT/'evaluation/rq2/ontology_grounding_strategies/all_onto_strategy_analysis.csv')))
for row in ontology_original:
    case_rows=[x for x in allrows if all(str(x[k])==str(row[key]) for k,key in
              [('model','llm'),('embedding','embedding'),('strategy','strategy')]) and 'k'+str(x['k'])==row['candidates']]
    N=len(case_rows);H=sum(x['hit'] for x in case_rows);C=sum(x['correct'] for x in case_rows)
    row['selection_accuracy']=C/H if H else 0;row['e2e_accuracy']=C/N if N else 0
export_at('evaluation/rq2/ontology_grounding_strategies/all_onto_strategy_analysis.csv',ontology_original)
# Keep per-configuration exports aligned, preserving their original column order.
for embedding in ('nomic','sapbert'):
    for strategy in ('term','avg','full'):
        for k in (10,20,50):
            subset=[{key:value for key,value in row.items() if key not in ('embedding','strategy','candidates')}
                    for row in ontology_original if row['embedding']==embedding and row['strategy']==strategy and row['candidates']=='k'+str(k)]
            export_at(f'evaluation/rq2/ontology_grounding_strategies/output/{embedding}_{strategy}_k{k}.csv',subset)
latex=DETAILS/'MI14_appendix_table.tex'
latex.write_text('\n'.join(line[:-4]+'\\' if line.endswith('\\\\') else line for line in latex.read_text().splitlines())+'\n')
# Emit data rows ready for insertion in the existing thesis tables.
labels={'mistral':'Mistral','mixtral_8x7':'Mixtral-8x7B','mixtral_8x22':'Mixtral-8x22B','llama':'Llama','phi4':'Phi-4'}
table_lines=[]
for section,records in [('ActivityDefinition grounding',ad_export),('Confidence',confidence_export),('Parameter identification',param_export),('Final encoding',encoding)]:
    table_lines.append('% '+section)
    for row in records:
        if section=='ActivityDefinition grounding':values=[f'{row[k]:.3f}' for k in ['recall_at_1','recall_at_2','recall_at_3','mrr','selection_accuracy','e2e_accuracy']]
        elif section=='Confidence':
            values=[str(row['correct_'+k]) for k in ['high','medium','low']]
            values += [f"{row['not_correct_'+k]} ({row['not_correct_'+k+'_no_cand']})" if row['not_correct_'+k] else '0' for k in ['high','medium','low']]
            values += [str(row[k]) for k in ['none_gt_in_cand','none_gt_not_in_cand']]
        elif section=='Parameter identification':values=[str(row['tp']+row['fn'])]+[str(row[k]) for k in ['tp','fp','fn']]+[f'{row[k]:.3f}' for k in ['precision','recall','f1-score']]
        else:values=[str(v) for k,v in row.items() if k!='llm']
        table_lines.append(labels[row['llm']]+' & '+' & '.join(values)+' \\')
    table_lines.append('')
(DETAILS/'thesis_table_rows.tex').write_text('\n'.join(table_lines))
print('Rebuilt RQ2/RQ3 tables and detailed audits from stored artifacts; no model inference.')

assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in INPUT_HASHES.items()), 'Source artifacts changed during regeneration'
manifest={'schema_version':1,'inputs_sha256':INPUT_HASHES,'models':models,
 'reference_action_unit':'One evaluable reference action; merged generated actions may contribute multiple correspondences',
 'rq3_comparison':'Ontology system/code with accepted reference alternatives; ValueSet code; decoded non-coded values; descriptive fields excluded for coded parameters',
 'source_artifacts_unchanged':True,'ontology_configurations_verified':len(onto),'ontology_parameter_configuration_cases':len(allrows),
 'rq3_reference_totals':{'temporal':26,'parameters':77}}
(DETAILS/'regeneration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
