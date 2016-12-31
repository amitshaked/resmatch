#!/usr/bin/env luajit
-- This file is copied from https://github.com/jzbonter/mc-cnn

require 'image'
require 'nn'
require 'cutorch'
require 'libadcensus'
require 'os'

cmd = torch.CmdLine()
cmd:option('-color', 'rgb')
cmd:option('-storage', 'storage')
opt = cmd:parse(arg)

for _, dataset in ipairs({2012, 2015}) do
   print(('dataset %d'):format(dataset))

   torch.manualSeed(42)
   if dataset == 2012 then
      n_disp = 40
      n_tr = 194
      n_te = 195
      data = opt.storage .. '/data.kitti'
      path = data .. '.' .. opt.color 
      image_0 = opt.colors == 1 and 'image_0' or 'colored_0'
      image_1 = opt.colors == 1 and 'image_1' or 'colored_1'
      disp_noc = 'disp_noc'
      nchannel = opt.color == 'rgb' and 3 or 1
   elseif dataset == 2015 then
      n_tr = 200
      n_disp = 40
      n_te = 200
      data = opt.storage .. '/data.kitti2015'
      path = data .. '.' .. opt.color 
      image_0 = 'image_2'
      image_1 = 'image_3'
      nchannel = opt.color == 'rgb' and 3 or 1
      disp_noc = 'disp_noc_0'
   end

	os.execute('mkdir -p ' .. path)

   height = 350
   width = 1242

   x0 = torch.FloatTensor(n_tr + n_te, nchannel, height, width):zero()
   x1 = torch.FloatTensor(n_tr + n_te, nchannel, height, width):zero()
   dispnoc = torch.FloatTensor(n_tr, 1, height, width):zero()
   metadata = torch.IntTensor(n_tr + n_te, 3):zero()

   examples = {}
   for i = 1,n_tr do
      examples[#examples + 1] = {dir='training', cnt=i}
   end

   for i = 1,n_te do
      examples[#examples + 1] = {dir='testing', cnt=i}
   end

   for i, arg in ipairs(examples) do
      img_path = '%s/unzip/%s/%s/%06d_10.png'
      img_0 = image.loadPNG(img_path:format(data, arg['dir'], image_0, arg['cnt'] - 1), nchannel, 'byte'):float()
      img_1 = image.loadPNG(img_path:format(data, arg['dir'], image_1, arg['cnt'] - 1), nchannel, 'byte'):float()

      if opt.colors == 1 and dataset == 2015 then
         img_0 = image.rgb2y(img_0)
         img_1 = image.rgb2y(img_1)
      end

      -- crop
      img_height = img_0:size(2)
      img_width = img_0:size(3)
      img_0 = img_0:narrow(2, img_height - height + 1, height)
      img_1 = img_1:narrow(2, img_height - height + 1, height)

      -- preprocess
      print(i)

      img_0:add(-img_0:mean()):div(img_0:std())
      img_1:add(-img_1:mean()):div(img_1:std())

      x0[{i,{},{},{1,img_width}}]:copy(img_0)
      x1[{i,{},{},{1,img_width}}]:copy(img_1)

      if arg['dir'] == 'training' then
         img_disp = torch.FloatTensor(1, img_height, img_width)
         adcensus.readPNG16(img_disp, ('%s/unzip/training/%s/%06d_10.png'):format(data, disp_noc, arg['cnt'] - 1))
         dispnoc[{i, 1}]:narrow(2, 1, img_width):copy(img_disp:narrow(2, img_height - height + 1, height))
      end

      metadata[{i, 1}] = img_height
      metadata[{i, 2}] = img_width
      metadata[{i, 3}] = arg['cnt'] - 1

      collectgarbage()
   end

   -- split train and test
   perm = torch.randperm(n_tr):long()
   te = perm[{{1,40}}]:clone()
   tr_disp = perm[{{41,41 + n_disp-1}}]:clone()
   tr = perm[{{41+ n_disp, n_tr}}]:clone()

   -- prepare tr dataset
   nnz_disp = torch.FloatTensor(23e6, 4)
   nnz_tr = torch.FloatTensor(23e6, 4)
   nnz_te = torch.FloatTensor(23e6, 4)
   nnz_disp_t = 0
   nnz_tr_t = 0
   nnz_te_t = 0
   for i = 1,n_tr do
      local disp = dispnoc[{{i}}]:cuda()
      adcensus.remove_nonvisible(disp)
      adcensus.remove_occluded(disp)
      adcensus.remove_white(x0[{{i}}]:cuda(), disp)
      disp = disp:float()

      is_te = false
      for j = 1,te:nElement() do
         if i == te[j] then
            is_te = true
         end
      end

      is_disp = false
      for j = 1,tr_disp:nElement() do
         if i == tr_disp[j] then
            is_disp = true
         end
      end
      if is_te then
         nnz_te_t = adcensus.make_dataset2(disp, nnz_te, i, nnz_te_t)
      elseif is_disp then
         nnz_disp_t = adcensus.make_dataset2(disp, nnz_disp, i, nnz_disp_t)
      else
         nnz_tr_t = adcensus.make_dataset2(disp, nnz_tr, i, nnz_tr_t)
      end
   end
   nnz_disp = torch.FloatTensor(nnz_disp_t, 4):copy(nnz_disp[{{1,nnz_disp_t}}])
   nnz_tr = torch.FloatTensor(nnz_tr_t, 4):copy(nnz_tr[{{1,nnz_tr_t}}])
   nnz_te = torch.FloatTensor(nnz_te_t, 4):copy(nnz_te[{{1,nnz_te_t}}])


   torch.save(('%s/x0.t7'):format(path), x0)
   torch.save(('%s/x1.t7'):format(path), x1)
   torch.save(('%s/dispnoc.t7'):format(path), dispnoc)
   torch.save(('%s/metadata.t7'):format(path), metadata)
   torch.save(('%s/tr_disp.t7'):format(path), tr_disp)
   torch.save(('%s/tr.t7'):format(path), tr)
   torch.save(('%s/te.t7'):format(path), te)
   torch.save(('%s/nnz_disp.t7'):format(path), nnz_disp)
   torch.save(('%s/nnz_tr.t7'):format(path), nnz_tr)
   torch.save(('%s/nnz_te.t7'):format(path), nnz_te)
end
