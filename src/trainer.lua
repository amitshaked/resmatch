require('optim')

local M = {}

local Trainer = torch.class('Trainer', M)

function Trainer:__init(network, ds_size, bs, optimState)
   self.bs = bs
   self.ds_size = ds_size
   self.network = network
   self.optim = optim.sgd
   self.optimState = optimState or {
      learningRate = network.params.lr,
      learningRateDecay = 0.0,
      momentum = network.params.mom,
      nesterov = true,
      dampening = 0.0,
      weightDecay = network.params.decay,
   }
end

function Trainer:train(epoch, trainBatch)

   self.optimState.learningRate = self.network:learningRate(epoch)

   x, dl_dx = self.network:getModelParameters()

   local function feval()
      return self.network:feval(x, dl_dx, self.inputs, self.targets)
   end

   local err_tr = 0
   local err_tr_cnt = 0
   local t = 1

   local indexes = torch.range(1, self.ds_size/self.bs):totable()
   local s = self.ds_size - self.bs
   for i, idx in ipairs(indexes) do
      xlua.progress(i,#indexes)
      t = (idx-1) * self.bs + 1
      self.inputs, self.targets = trainBatch(t, self.bs, self.network.params.ws)

      _, fs = self.optim(feval, x, self.optimState)
      local err = fs[1]
      if err >= 0 and err < 100 then
         err_tr = err_tr + err
         err_tr_cnt = err_tr_cnt + 1
      else
         print(('WARNING! err=%f'):format(err))
         if err ~= err then
            os.exit()
         end
      end
   end
   xlua.progress(#indexes, #indexes)
   return err_tr / err_tr_cnt
end

return M.Trainer
