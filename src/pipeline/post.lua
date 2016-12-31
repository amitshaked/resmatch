local M = {}

function M.process(vols, x_batch, disp_max, params, dataset, sm_terminate, sm_skip , directions)

   local vol

   for _, direction in ipairs(directions) do
       vol = vols[{{direction == -1 and 1 or 2}}]

       sm_active = (sm_terminate ~= 'cnn')

       -- cross computation
       local x0c, x1c
		 --print(sm_skip)
       if sm_active and sm_skip ~= 'cbca' then
           x0c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
           x1c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
           adcensus.cross(x_batch[1], x0c, params.L1, params.tau1)
           adcensus.cross(x_batch[2], x1c, params.L1, params.tau1)
           local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
           for i = 1,params.cbca_i1 do
               adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
               vol:copy(tmp_cbca)
           end
           tmp_cbca = nil
           collectgarbage()
       end
       sm_active = sm_active and (sm_terminate ~= 'cbca1')

       if sm_active and sm_skip ~= 'sgm' then
           vol = vol:transpose(2, 3):transpose(3, 4):clone()
           collectgarbage()
           do
               local out = torch.CudaTensor(1, vol:size(2), vol:size(3), vol:size(4))
               local tmp = torch.CudaTensor(vol:size(3), vol:size(4))
               for _ = 1,params.sgm_i do
                   out:zero()
                   adcensus.sgm2(x_batch[1], x_batch[2], vol, out, tmp, params.pi1, params.pi2, params.tau_so,
                   params.alpha1, params.sgm_q1, params.sgm_q2, direction)
                   vol:copy(out):div(4)
               end
               vol:resize(1, disp_max, x_batch:size(3), x_batch:size(4))
               vol:copy(out:transpose(3, 4):transpose(2, 3)):div(4)

           end
           collectgarbage()
       end
       sm_active = sm_active and (sm_terminate ~= 'sgm')

       if sm_active and sm_skip ~= 'cbca' then
           local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
           for i = 1,params.cbca_i2 do
               adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
               vol:copy(tmp_cbca)
           end
       end
       sm_active = sm_active and (sm_terminate ~= 'cbca2')
       vols[{{direction == -1 and 1 or 2}}] = vol
   end
	return vols
end

return M
