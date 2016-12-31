
local M = {}

local Runner = torch.class('Runner', M)

function Runner:__init(mcnet, gdn, dataset, opt)
   self.matcher = require('pipeline/matching')
   self.disp = require('pipeline/disparity')
   self.post = require('pipeline/post')
   self.refiner = require('pipeline/refinement')
   self.dataset = dataset
   self.mcnet = mcnet
   self.gdn = gdn
   self.opt = opt
   self.path = ('%s/cache/%s/%s'):format(opt.storage, self.dataset.name, self.mcnet.name)
end

function Runner:setGdn(gdn)
   self.gdn = gdn
end

function Runner:predict(img, disp_max, directions, make_cache)

   local vox
   -- compute matching cost
   if self.opt.use_cache then
      vox = torch.load(('%s_%s.t7'):format(self.path, img.id)):cuda()
   else
      vox = self.matcher.match(self.mcnet, img.x_batch,
         disp_max, directions):cuda()
      if make_cache then
         torch.save(('%s_%s.t7'):format(self.path, img.id), vox)
      end
   end
   collectgarbage()

   -- post_process
   vox = self.post.process(vox, img.x_batch, disp_max, self.mcnet.params, self.dataset, self.opt.sm_terminate, self.opt.sm_skip, directions)

   -- pred after post process
   local vox_simple = vox:clone()

   -- disparity image
   local disp, vox, conf, t = self.disp.disparityImage(vox, self.gdn)

   -- refinement
   disp = self.refiner.refine(disp, vox_simple, self.mcnet.params, self.dataset, self.opt.sm_skip ,self.opt.sm_terminate, disp_max, conf, t.t1, t.t2)

   return disp[2]

end

function Runner:test(range, showall, make_cache)
   local err_sum = 0

   local opt = self.opt
   local directions = self.dataset.name == 'mb' and {-1} or {1, -1}

   for i, idx in ipairs(range) do
      xlua.progress(i-1, #range)
      local img = self.dataset:getTestSample(idx, false)
      local disp_max = img.disp_max or self.dataset.disp_max

      cutorch.synchronize()
      sys.tic()

      local pred = self:predict(img, disp_max, directions)

      cutorch.synchronize()
      local runtime = sys.toc()
      assert(pred:sum() == pred:sum())

      local dispnoc = img.dispnoc
      local mask = torch.CudaTensor(dispnoc:size()):ne(dispnoc, 0)

      err, pred_bad, pred_good = self:calcErr(pred, dispnoc:clone(), mask, self.dataset.err_at)
      err_sum = err_sum + err

      if showall then
         print('\n' .. img.id, runtime, err .. '\n')
      end
      if self.opt.save_img then
         save_png(self.dataset, img, disp_max, pred, pred_bad, pred_good, mask)
      end
   end
   xlua.progress(#range, #range)
   return err_sum / #range
end

function Runner:submit(samples)
   os.execute('rm -rf out/*')
   if self.dataset.name == 'kitti2015' then
      os.execute('mkdir out/disp_0')
   end

   local directions = self.dataset.name == 'mb' and {-1} or {1, -1}
   for i, idx in ipairs(samples) do
      xlua.progress(i, #samples)

      local img = self.dataset:getTestSample(idx, true)
      local disp_max = img.disp_max or self.dataset.disp_max
      local pred = self:predict(img, disp_max, directions)

      if self.dataset.name == 'kitti' or self.dataset.name == 'kitti2015' then
         local pred_img = torch.FloatTensor(img.height, img.width):zero()
         pred_img:narrow(1, img.height - self.dataset.height + 1, self.dataset.height):copy(pred[{1,1}])

         if self.dataset.name == 'kitti' then
            path = 'out'
         elseif self.dataset.name == 'kitti2015' then
            path = 'out/disp_0'
         end
         local s = ("%s/%06d_10.png"):format(path, img.id)
         adcensus.writePNG16(pred_img, img.height, img.width, s)
      elseif self.dataset.name == 'mb' then
         local base = 'out/' .. self.dataset.fname_submit[img.id - (#self.dataset.X - #self.dataset.fname_submit)]
         os.execute('mkdir -p ' .. base)
         adcensus.writePFM(image.vflip(pred[{1,1}]:float()), base .. '/disp0' .. opt.METHOD_NAME .. '.pfm')
         local f = io.open(base .. '/time' .. opt.METHOD_NAME .. '.txt', 'w')
         f:write(tostring(runtime))
         f:close()
      end
   end
   os.execute('cd out; zip -r submission.zip .')
end

function Runner:createDispData()

   local range = self.dataset:getDispRange(self.opt)
   local samples = torch.FloatTensor(#range, self.dataset.disp_max, self.dataset.X0:size(3), self.dataset.X0:size(4))
   local indexes = {}
   local directions = self.dataset.name == 'mb' and {-1} or {1, -1}
   for j, i in ipairs(range) do
      -- Get the sample to prepare
      local img = self.dataset:getTestSample(i)
      local disp_max = img.disp_max or self.dataset.disp_max

      -- 2 directions for left-right consistency check

      -- Compute the matching cost map
      vox = self.mcnet:computeMatchingCost(img.x_batch, self.dataset.disp_max,directions):cuda()

      -- Post processing
      vox = self.post.process(vox, img.x_batch, disp_max, self.mcnet.params, self.dataset, '','', directions)
      vox = nn.Tanh():cuda():forward(vox)

      -- Matching cost to similarity score
      vox:mul(-1):add(1)
      samples[{{j}, {}, {}, {1, vox:size(4)}}] = vox[{{1}}]:float()
      indexes[i] = j

      xlua.progress(j, #range)
   end
   return samples, indexes
end
function save_png(dataset, img, disp_max, pred, pred_bad, pred_good, mask)

   local img_pred = torch.Tensor(1, 3, pred:size(3), pred:size(4))
   adcensus.grey2jet(pred:double():add(1)[{1,1}]:div(disp_max):double(), img_pred)
   local x0 = img.x_batch[1]
   if x0:size(1) == 1 then
      x0 = torch.repeatTensor(x0:cuda(), 3, 1, 1)
   end
   img_err = x0:mul(50):add(150):div(255)

   local real = torch.CudaTensor():resizeAs(img_err):copy(img_err)
   img_err[{1}]:add( 0.7, pred_bad)
   img_err[{2}]:add(-0.7, pred_bad)
   img_err[{3}]:add(-0.7, pred_bad)
   img_err[{1}]:add(-0.7, pred_good)
   img_err[{2}]:add( 0.7, pred_good)
   img_err[{3}]:add(-0.7, pred_good)

   local gt
   if dataset.name == 'kitti' or dataset.name == 'kitti2015' then
      gt = img.dispnoc
   elseif dataset.name == 'mb' then
      gt = img.dispnoc:resize(1, 1, pred:size(3), pred:size(4))
   end
   local img_gt = torch.Tensor(1, 3, pred:size(3), pred:size(4)):zero()
   adcensus.grey2jet(gt:double():add(1)[{1}]:div(disp_max):double(), img_gt)
   img_gt[{1,3}]:cmul(mask:double())

   image.save(('tmp/%s_%s_gt.png'):format(dataset.name, img.id), img_gt[1])
   image.save(('tmp/%s_%s_real.png'):format(dataset.name, img.id), real[1])
   image.save(('tmp/%s_%s_%s_pred.png'):format(dataset.name, network.name, img.id), img_pred[1])
   image.save(('tmp/%s_%s_%s_err.png'):format(dataset.name, network.name, img.id), img_err[1])
end

function Runner:calcErr(pred, dispnoc, mask, err_at)
   local pred_good = torch.CudaTensor(dispnoc:size())
   local pred_bad = torch.CudaTensor(dispnoc:size())
   dispnoc:add(-1, pred):abs()
   pred_bad:gt(dispnoc, err_at):cmul(mask)
   pred_good:le(dispnoc, self.dataset.err_at):cmul(mask)

   local err = pred_bad:sum() / mask:sum()

   return err, pred_bad, pred_good
end

return M.Runner
