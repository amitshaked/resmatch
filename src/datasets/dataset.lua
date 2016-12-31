local M = {}

local Dataset = torch.class('Dataset', M)

function Dataset:__init(self, opt)
   torch.manualSeed(opt.seed)
   cutorch.manualSeed(opt.seed)
   self.__index = self
   self.n_colors = opt.color == 'rgb' and 3 or 1
   self:setParams()
   self:load(opt)
   if opt.a == 'train_mcn' or opt.a == 'train_gdn' then
      self:prepareTrainingData(opt.subset, opt.all)
   end
end


function Dataset:obfuscationParams()
   assert(self.hscale <= 1 and self.scale <= 1)

   local params = {}
   params.x0 = {}
   params.x1 = {}
   local s = torch.uniform(self.scale, 1)
   params.x0.scale = {s * torch.uniform(self.hscale, 1), s}
   if self.hflip == 1 and torch.uniform() < 0.5 then
      params.x0.scale[1] = -params.x0.scale[1]
   end
   if self.vflip == 1 and torch.uniform() < 0.5 then
      params.x0.scale[2] = -params.x0.scale[2]
   end
   params.x0.hshear = torch.uniform(-self.hshear, self.hshear)
   params.x0.trans = {torch.uniform(-self.trans, self.trans), torch.uniform(-self.trans, self.trans)}
   params.x0.phi = torch.uniform(-self.rotate * math.pi / 180, self.rotate * math.pi / 180)
   params.x0.brightness = torch.uniform(-self.brightness, self.brightness)

   assert(self.contrast >= 1 and self.d_contrast >= 1)
   params.x0.contrast = torch.uniform(1 / self.contrast, self.contrast)

   params.x1.scale = {params.x0.scale[1] * torch.uniform(self.d_hscale, 1), params.x0.scale[2]}
   params.x1.hshear = params.x0.hshear + torch.uniform(-self.d_hshear, self.d_hshear)
   params.x1.trans = {params.x0.trans[1], params.x0.trans[2] + torch.uniform(-self.d_vtrans, self.d_vtrans)}
   params.x1.phi = params.x0.phi + torch.uniform(-self.d_rotate * math.pi / 180, self.d_rotate * math.pi / 180)
   params.x1.brightness = params.x0.brightness + torch.uniform(-self.d_brightness, self.d_brightness)
   params.x1.contrast = params.x0.contrast * torch.uniform(1 / self.d_contrast, self.d_contrast)

   return params
end

local function mul32(a,b)
   return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3], a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
end

function Dataset:makePatch(src, dst, dim3, dim4, ws, params)
   local m = {1, 0, -dim4, 0, 1, -dim3}

   if params then
      m = mul32({1, 0, params.trans[1], 0, 1, params.trans[2]}, m) -- translate
      m = mul32({params.scale[1], 0, 0, 0, params.scale[2], 0}, m) -- scale
      local c = math.cos(params.phi)
      local s = math.sin(params.phi)
      m = mul32({c, s, 0, -s, c, 0}, m) -- rotate
      m = mul32({1, params.hshear, 0, 0, 1, 0}, m) -- shear
   end

   m = mul32({1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2}, m)
   m = torch.FloatTensor(m)
   cv.warp_affine(src, dst, m)
   if params then
      dst:mul(params.contrast):add(params.brightness)
   end
end


function Dataset:prepareTrainingData(subset, all)
   self.nnz_tr = torch.cat(self.nnz_tr, self.nnz_disp, 1)
   self.tr = torch.cat(self.tr, self.tr_disp, 1)
   self.nnz_disp = self.nnz_tr
   self.tr_disp = self.tr
   -- subset training dataset
   if subset < 1 then
      self.nnz = self:subset(self.nnz_tr, self.tr, subset)
      self.disp = self:subset(self.nnz_disp, self.tr_disp, subset)
   elseif all then
      self.nnz = torch.cat(self.nnz_tr, self.nnz_te, 1)   
      self.disp = torch.cat(self.nnz_disp, self.nnz_te, 1)
   else
      self.nnz = self.nnz_tr
      self.disp = self.nnz_disp
   end

   collectgarbage()
end

function Dataset:shuffle()
   self.perm = torch.randperm(self.nnz:size(1))
   self.perm_disp = torch.randperm(self.disp:size(1))
end

function Dataset:getDispRange(opt)
   -- Get disparity samples range
   local range
   if opt.all then 
      range =  torch.totable(torch.cat(self.tr_disp, self.te))
   else
      range = torch.totable(self.tr_disp)
   end
   return range
end

function Dataset:saveDispData(samples, indexes, mcname)
   local path = ('%s/disparity/%s'):format(self.dir, mcname)
   torch.save(path .. '.t7', samples)
   torch.save(path .. '.indexes', indexes)
   self.X2 = samples
   self.X2_idx = indexes
end


function Dataset:loadDispData(mcname)
   local time = sys.clock()
   local path = ('%s/disparity/%s'):format(self.dir, mcname)

   if paths.filep(path .. '.t7') then
      print('===> Loading disparity training set...')
      self.X2 = self.X2 or torch.load(path .. '.t7')
      self.X2_idx = self.X2_idx or torch.load(path ..'.indexes')
      print(('===> Loaded! time=%s'):format(sys.clock() - time))
      return true
   else
      print('===> No training data found for disparity network')
      return false
   end
end

function Dataset:trainingSamples(start, size, ws)
   local x = torch.FloatTensor(size * 4, self.n_colors, ws, ws)
   local y = torch.FloatTensor(size * 2)

   for i=start, start+size-1 do
      local idx = self.perm[i]
      local img = self.nnz[{idx, 1}]
      local dim3 = self.nnz[{idx, 2}]
      local dim4 = self.nnz[{idx, 3}]
      local d = self.nnz[{idx, 4}]

      local d_pos = torch.uniform(-self.true1, self.true1)
      local d_neg = torch.uniform(self.false1, self.false2)
      if torch.uniform() < 0.5 then
         d_neg = -d_neg
      end
      local x0, x1 = self:getLR(img)

      idx = i-start+1
      local params = self:obfuscationParams()
      self:makePatch(x0, x[idx * 4 - 3], dim3, dim4, ws, params.x0)
      self:makePatch(x1, x[idx * 4 - 2], dim3, dim4 - d + d_pos, ws, params.x1)
      self:makePatch(x0, x[idx * 4 - 1], dim3, dim4, ws, params.x0)
      self:makePatch(x1, x[idx * 4 - 0], dim3, dim4 - (d-d_neg), ws, params.x1)

      y[idx * 2 - 1] = 0
      y[idx * 2] = 1
   end

   return x:cuda(), y:cuda()
end


local function scale(input, size)

   local temp = torch.FloatTensor(input:size(1), size, size)
   image.scale(temp, input, 'bilinear')
   return temp
end

function Dataset.sample(xs, p)
   local perm = torch.randperm(xs:nElement()):long()
   return xs:index(1, perm[{{1, xs:size(1) * p}}])
end

return M.Dataset