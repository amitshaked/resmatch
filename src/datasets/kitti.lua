Dataset = require('datasets/dataset')

local KittiDataset, parent = torch.class('KittiDataset', 'Dataset')

local function createDataset(opt)
   return KittiDataset:new(opt)
end

function KittiDataset:__init(self, opt)
   self.name = 'kitti'
   self.kittiDir = opt.storage .. '/data.kitti.' .. opt.color
   self.kitti2015Dir = opt.storage .. '/data.kitti2015.' .. opt.color
   self.dir = self.kittiDir
   parent.__init(parent, self, opt)
end

function KittiDataset:setParams()
   -- parameters for training
   self.true1 = 1
   self.false1 = 4
   self.false2 = 10
   -- parameters for image transformations
   self.hflip = 0
   self.vflip = 0
   self.rotate = 7
   self.hscale = 0.9
   self.scale = 1
   self.trans = 0
   self.hshear = 0.1
   self.brightness = 0.7
   self.contrast = 1.3
   self.d_vtrans = 0
   self.d_rotate = 0
   self.d_hscale = 1
   self.d_hshear = 0
   self.d_brightness = 0.3
   self.d_contrast = 1

   --parameters for the network
   self.height = 350
   self.width = 1242
   self.disp_max = 228
   self.n_te = 195
   self.n_tr = 194
   self.err_at = 3

end

function KittiDataset:load(opt)
   if not opt.mix then
      self:load_data()
   else
      function load(fname)
         local X_12 = torch.load(self.kittiDir .. '/' .. fname)
         local X_15 = torch.load(self.kitti2015Dir .. '/' .. fname)
         local X = torch.cat(X_12[{{1,194}}], X_15[{{1,200}}], 1)
         X = torch.cat(X, dataset == 'kitti' and X_12[{{195,389}}] or X_15[{{200,400}}], 1)
         return X
      end

      self.X0 = load('x0.t7')
      self.X1 = load('x1.t7')
      self.metadata = load('metadata.t7')

      self.dispnoc = torch.cat(torch.load(opt.storage .. self.kittiDir .. '/dispnoc.t7'), torch.load(opt.storage .. self.kitti2015Dir .. '/dispnoc.t7'), 1)
      self.tr = torch.cat(torch.load(opt.storage .. self.kittiDir .. '/tr.t7'), torch.load(opt.storage .. self.kitti2015Dir .. '/tr.t7'):add(194))
      self.te = self.name == 'kitti' and torch.load(self.kittiDir .. '/te.t7') or torch.load(self.kitti2015Dir .. '/te.t7'):add(194)
      function load_nnz(fname)
         local X_12 = torch.load(opt.storage .. self.kittiDir .. '/' .. fname)
         local X_15 = torch.load(opt.storage .. self.kitti2015Dir .. '/' .. fname)
         X_15[{{},1}]:add(194)
         return torch.cat(X_12, X_15, 1)
      end

      self.nnz_tr = load_nnz('nnz_tr.t7')
      self.nnz_te = load_nnz('nnz_te.t7')
   end
end

function KittiDataset:load_data()

   self.X0 = torch.load(('%s/x0.t7'):format(self.dir))
   self.X1 = torch.load(('%s/x1.t7'):format(self.dir))
   self.dispnoc = torch.load(('%s/dispnoc.t7'):format(self.dir))
   self.metadata = torch.load(('%s/metadata.t7'):format(self.dir))
   self.tr_disp = torch.load(('%s/tr_disp.t7'):format(self.dir))
   self.tr = torch.load(('%s/tr.t7'):format(self.dir))
   self.te = torch.load(('%s/te.t7'):format(self.dir))
   self.nnz_disp = torch.load(('%s/nnz_disp.t7'):format(self.dir))
   self.nnz_tr = torch.load(('%s/nnz_tr.t7'):format(self.dir))
   self.nnz_te = torch.load(('%s/nnz_te.t7'):format(self.dir))
end

function KittiDataset:subset(ds, tr, subset)
   local tr_subset = Dataset.sample(tr, subset)
   local nnz_tr_output = torch.FloatTensor(ds:size()):zero()
   local t = adcensus.subset_dataset(tr_subset, ds, nnz_tr_output);

   return nnz_tr_output[{{1,t}}]
end


function KittiDataset:getTestSample(i, submit)
   local img = {}

   img.height = self.metadata[{i,1}]
   img.width = self.metadata[{i,2}]

   img.id = self.metadata[{i,3}]
   if not submit then
      img.dispnoc = self.dispnoc[{i,{},{},{1,img.width}}]:cuda()
   end
   x0 = self.X0[{{i},{},{},{1,img.width}}]
   x1 = self.X1[{{i},{},{},{1,img.width}}]

   img.x_batch = torch.CudaTensor(2, self.n_colors, self.height, self.width)
   img.x_batch:resize(2, self.n_colors, x0:size(3), x0:size(4))
   img.x_batch[1]:copy(x0)
   img.x_batch[2]:copy(x1)

   return img
end

function KittiDataset:getSubmissionRange()
   return  torch.totable(torch.range(self.X0:size(1) - self.n_te + 1, self.X0:size(1)))

end

function KittiDataset:getTestRange()
   return torch.totable(self.te)
end

function KittiDataset:getLR(img)
   local x0 = self.X0[img]
   local x1 = self.X1[img]
   return x0, x1
end

return createDataset
