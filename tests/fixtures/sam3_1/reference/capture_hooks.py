# Capture-only Python return profiler; upstream source and tensor math remain unchanged.
_capture_counts={}
def _capture_return(frame,event,result):
 if event!='return':return
 name=frame.f_code.co_name;file=frame.f_code.co_filename
 if name not in ('_forward_sam_heads','_postprocess','nms'):return
 if '/sam3' not in file:return
 number=_capture_counts.get(name,0);_capture_counts[name]=number+1
 loc=frame.f_locals; tensors={}; meta={}
 if name=='_forward_sam_heads':
  for key in ['best_iou_inds','ious','sam_output_tokens']:
   value=loc.get(key)
   if isinstance(value,mx.array):tensors[key]=value
  if isinstance(result,dict):
   for key in ['ious','low_res_masks','high_res_masks','object_score_logits']:
    value=result.get(key)
    if isinstance(value,mx.array):tensors['out_'+key]=value
 elif name=='_postprocess':
  for key in ['keep','scores']:
   if key in loc:meta[key]=np.asarray(loc[key]).tolist()
  for key in ['pred_logits','pred_boxes','pred_masks','presence_logits']:
   value=loc.get('outputs',{}).get(key)
   if isinstance(value,mx.array):tensors[key]=value
 elif name=='nms':
  for key in ['keep','order','indices']:
   if key in loc:meta[key]=np.asarray(loc[key]).tolist()
 if tensors:
  mx.eval(tensors);mx.save_safetensors(str(O/f'capture-{name}-{number}.safetensors'),tensors)
 if meta:(O/f'capture-{name}-{number}.json').write_text(json.dumps(meta,indent=2)+'\n')
sys.setprofile(_capture_return)
