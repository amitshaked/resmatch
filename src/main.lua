#!/usr/local/bin/lua luajit
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
require 'torch'
require 'image'
require 'paths'

require '../libadcensus'
require '../libcv'
require '../libcuresmatch'


-- Initialize components
local opts = require 'opts'
local opt = opts.parse(arg)
local log = require('logger')(opt)
local dataset = require('datasets/' .. opt.ds)(opt)
local mcnet = require('networks/mc-models/'..opt.mc..
   '/'..opt.m):new(opt, dataset)
local runner = require('runner')(mcnet, nil, dataset, opt)
local Trainer = require('trainer')

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

-- Set the actions to do, optins are:
--    train_mcn   - train the matching cost network
--    train_gdn   - train the global disparity network
--    test        - test the pipeline with validation data
--    submit      - create the submission file for the dataset's
--                  online evaluation server
pipeline = {}
pipeline[opt.a] = true
if opt.a == 'train_mcn' then
   if opt.all then
      pipeline['train_gdn'] = true
      pipeline['submit'] = true
   end
elseif opt.a == 'train_gdn' then
   pipeline['test'] = true
   if opt.all then
      pipeline['submit'] = true
   end
end

-- Load last checkpoint if exists
print('===> Loading matching cost network...')
local checkpoint, optimState = mcnet:load(opt)
print('===> Loaded! Network: ' .. mcnet.name)

-- Training the matching cost network
if pipeline['train_mcn'] then

   local start_epoch = checkpoint and checkpoint.epoch +1 or opt.start_epoch

   -- Initialize new trainer for the MCN
   local trainer = Trainer(mcnet, dataset.nnz:size(1), mcnet.params.bs/2, optimState)

   -- The function the trainer uses to get the next batch
   local function trainingBatch(start, size, ws)
      return dataset:trainingSamples(start,size,ws)
   end

   print('===> training matching cost network') 
   for epoch = start_epoch, opt.epochs do
      dataset:shuffle() -- to get random order of samples

      -- Train one epoch of all the samples
      local err_tr = trainer:train(epoch, trainingBatch)

      -- Output results
      local msg = ('train epoch %g\t err %g\tlr %g\n')
         :format(epoch, err_tr, trainer.optimState.learningRate)
      log:write(msg)

      -- Save the current checkpoint
      mcnet:save(epoch, optimState)

      -- Run validation if wanted
      local validate = ((opt.debug and epoch % opt.times == 0)
         or (epoch >= opt.after)) and epoch < opt.epochs
      if validate then
         print('===> testing...')
         local err_te = runner:test(dataset:getTestRange(), false, false)

         -- Output validation results
         log:write(('test epoch: %g\terror: %g\n'):format(epoch, err_te))
      end
   end

   -- After training is completed test and save the final model
   mcnet:save(0, optimState)
   local err_te = runner:test(dataset:getTestRange(), true, opt.make_cache)
   log:write(err_te)
end

-- Train the global disparity network
local dnet
if opt.gdn ~= '' then

   dnet = require('networks/gdn-models/' .. opt.gdn)(opt, dataset, mcnet.name)
   runner:setGdn(dnet)

   if pipeline['train_gdn'] then

      -- Load disparity data
      local ok = dataset:loadDispData(mcnet.name)

		-- If non exists create and save it
		if not ok then
      	print('===> Creating training data for network ' .. mcnet.name)
			local samples, indexes = runner:createDispData()
			dataset:saveDispData(samples, indexes, mcnet.name)
		end

      -- Load last checkpoint if exists
      checkpoint, optimState = dnet:load(opt)
      local start_epoch = checkpoint and checkpoint.epoch +1 or opt.start_epoch


      -- Initialize new trainer for the GDN
      local trainer = Trainer(dnet, dataset.disp:size(1), dnet.params.bs/2, optimState)

      -- The function the trainer uses to get the next batch
      local function getTrainBatch(start, size, ws)
         return dnet:getDisparityTrainingSamples(start,size,ws)
      end

      print('===> training disparity network...')
		print('===> Starting from epoch ' .. start_epoch)
      for epoch = start_epoch, opt.epochs do
         dataset:shuffle() -- samples in random order

         local err_tr = trainer:train(epoch, getTrainBatch)

         -- Output results
         local msg = ('train epoch %g\t err %g\tlr %g\n'):format(epoch, err_tr, trainer.optimState.learningRate)
         log:write(msg)

         -- Save the current checkpoint
         dnet:save(epoch, optimState)

         -- Run validation if wanted
         local validate = ((opt.debug and epoch % opt.times == 0) or (epoch >= opt.after)) and epoch < opt.epochs
         if validate then
            print('===> testing...')
            local err_te = runner:test(dataset:getTestRange(), false, false)

            -- Output validation results
            log:write(('test: %g\t%g\n'):format(epoch, err_te))
         end
      end

      -- After training is completed test and save the final model
      dnet:save(0, optimState)
      local err_te = runner:test(dataset:getTestRange(), true, false)
      log:write(err_te)
   else
      dnet:load(opt)
   end
end

-- Run validation
if pipeline['test'] then
   local err_te = runner:test(dataset:getTestRange(), true, opt.make_cache)
   log:write(err_te)
end

-- Submit results
if pipeline['submit'] then
   submission_range = dataset:getSubmissionRange()
   runner:submit(submission_range)
end
