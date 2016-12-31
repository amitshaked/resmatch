local M = {}

function M.refine(disp, vols, opt, dataset, sm_skip, sm_terminate, disp_max, conf, t1, t2)
   local sm_active = true
   if dataset.name == 'kitti' or dataset.name == 'kitti2015' then
       local outlier = torch.CudaTensor():resizeAs(disp[2]):zero()
       curesmatch.outlier_detection(disp[2], disp[1], outlier, disp_max, conf[1], conf[2], t1, t2)

       if sm_active and sm_skip ~= 'occlusion' then

           disp[2] = adcensus.interpolate_occlusion(disp[2], outlier)

       end
       sm_active = sm_active and (sm_terminate ~= 'occlusion')

       if sm_active and sm_skip ~= 'mismatch' then 
           disp[2] = adcensus.interpolate_mismatch(disp[2], outlier)

       end
       sm_active = sm_active and (sm_terminate ~= 'mismatch')
   end
   if sm_active and sm_skip ~= 'subpixel_enhancement' then
       disp[2] = adcensus.subpixel_enchancement(disp[2], vols[{{1}}], disp_max)

   end
   sm_active = sm_active and (sm_terminate ~= 'subpixel_enchancement')

   if sm_active and sm_skip ~= 'median' then
       disp[2] = adcensus.median2d(disp[2], 5)

   end
   sm_active = sm_active and (sm_terminate ~= 'median')

   if sm_active and sm_skip ~= 'bilateral' then
       disp[2] = adcensus.mean2d(disp[2], gaussian(opt.blur_sigma):cuda(), opt.blur_t)

   end

   return disp

end

function gaussian(sigma)
   local kr = math.ceil(sigma * 3)
   local ks = kr * 2 + 1
   local k = torch.Tensor(ks, ks)
   for i = 1, ks do
      for j = 1, ks do
         local y = (i - 1) - kr
         local x = (j - 1) - kr
         k[{i,j}] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
      end
   end
   return k
end
return M
