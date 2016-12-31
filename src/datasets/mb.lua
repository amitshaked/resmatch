Dataset = require('dataset')

local MbDataset, parent = torch.class('MbDataset',Dataset)

function MbDataset:__init(self, opt)
   parent.__init(parent, self, opt)
   self.name='mb'
end

function MbDataset:setParams()
   --parameters for training
   self.true1 = 0.5
   self.false1 = 1.5
   self.false2 = 6

   -- parameters for image transformations
   self.hflip = 0
   self.vflip=  0
   self.rotate = 28
   self.hscale = 0.8
   self.scale = 0.8
   self.trans = 0
   self.hshear = 0.1
   self.brightness = 1.3
   self.contrast = 1.1
   self.d_vtrans = 1
   self.d_rotate = 3
   self.d_hscale = 0.9
   self.d_hshear = 0.3
   self.d_brightness = 0.7
   self.d_contrast = 1.1

   self.d_light=0.2
   self.d_exp=0.2
   --parameters for the network
   self.rect = opt.rect
   self.n_colors = opt.color == 'rgb' and 3 or 1
   self.color = opt.color

   self.height = 1500
   self.width = 1000
   self.disp_max = 200

   self.err_at = 1


end

function MbDataset:load(opt)  
   local data_dir = ('%s/data.mb.%s_%s'):format(opt.storage, self.rect, self.color)
   self.te = fromfile(('%s/te.bin'):format(data_dir))
   self.metadata = fromfile(('%s/meta.bin'):format(data_dir))
   self.nnz_tr = fromfile(('%s/nnz_tr.bin'):format(data_dir))
   self.nnz_te = fromfile(('%s/nnz_te.bin'):format(data_dir))
   self.fname_submit = {}
   for line in io.open(('%s/fname_submit.txt'):format(data_dir), 'r'):lines() do
      table.insert(self.fname_submit, line)
   end
   self.X = {}
   self.dispnoc = {}
   local fname = ""
   for n = 1,self.metadata:size(1) do
      local XX = {}
      local light = 1
      while true do
         fname = ('%s/x_%d_%d.bin'):format(data_dir, n, light)
         if not paths.filep(fname) then
            break
         end
         table.insert(XX, fromfile(fname))
         light = light + 1
         if opt.a == 'test_te' or opt.a == 'submit' then
            break  -- we don't need to load training data
         end
      end
      table.insert(self.X, XX)

      fname = ('%s/dispnoc%d.bin'):format(data_dir, n)
      if paths.filep(fname) then
         table.insert(self.dispnoc, fromfile(fname))
      end
   end
end

function MbDataset:subset(ds,tr, subset)
   local tr_2014 = sample(torch.range(11, 23):long(), subset)
   local tr_2006 = sample(torch.range(24, 44):long(), subset)
   local tr_2005 = sample(torch.range(45, 50):long(), subset)
   local tr_2003 = sample(torch.range(51, 52):long(), subset)
   local tr_2001 = sample(torch.range(53, 60):long(), subset)

   local tr_subset = torch.cat(tr_2014, tr_2006)
   tr_subset = torch.cat(tr_subset, tr_2005)
   tr_subset = torch.cat(tr_subset, tr_2003)
   tr_subset = torch.cat(tr_subset, tr_2001)

   local nnz_tr_output = torch.FloatTensor(ds:size()):zero()
   local t = adcensus.subset_dataset(tr_subset, ds, nnz_tr_output);
   return nnz_tr_output[{{1,t}}]

end


function MbDataset:getSubmissionRange()
   local examples = {}
   -- for i = #X - 14, #X do
   for i = #self.X - 29, #self.X do
      table.insert(examples, {i, 2})
   end
   return examples
end

function MbDataset:getTestTange()
   local examples = {}
   for i = 1,self.te:nElement() do
      table.insert(examples, {self.te[i], 2})
   end
   table.insert(examples, {5, 3})
   table.insert(examples, {5, 4})
   return examples
end

function MbDataset:getTestSample(i)
   local img = {}

   local i, right = table.unpack(i)
   img.id = ('%d_%d'):format(i, right)
   img.disp_max = self.metadata[{i,3}]
   local x0 = self.X[i][1][{{1}}]
   local x1 = self.X[i][1][{{right}}]
   img.x_batch = torch.CudaTensor(2, self.n_colors, self.height, self.width)
   img.x_batch:resize(2, self.n_colors, x0:size(3), x0:size(4))
   --print(img.x_batch:size(), x0:size())
   img.x_batch[1]:copy(x0)
   img.x_batch[2]:copy(x1)

   img.dispnoc = self.dispnoc[i]:cuda()
   return img
end

function MbDataset:prepareTrainingData(subset, action, all)
   -- subset training dataset
   if subset < 1 then
      self.nnz = self:subset(self.nnz_tr, self.tr, subset)
   elseif all then
      self.nnz = torch.cat(self.nnz_tr, self.nnz_te, 1)
   else
      self.nnz = self.nnz_tr
   end
   self.disp = self.nnz
   self.nnz_disp = self.nnz_tr
end

function MbDataset:getLR(img)
   local x0, x1
   local light = (torch.random() % (#self.X[img] - 1)) + 2
   local exp = (torch.random() % self.X[img][light]:size(1)) + 1
   local light_ = light
   local exp_ = exp
   if torch.uniform() < self.d_exp then
      exp_ = (torch.random() % self.X[img][light]:size(1)) + 1
   end
   if torch.uniform() < self.d_light then
      light_ = math.max(2, light - 1)
   end
   x0 = self.X[img][light][{exp,1}]
   x1 = self.X[img][light_][{exp_,2}]
   return x0, x1
end
