
local M = {}

function M.parse (arg)

   local cmd = torch.CmdLine()
   cmd:option('-ds', 'kitti', 'Dataset')
   cmd:option('-mc', 'resmatch', 'Matching cost network architecture')
   cmd:option('-m', 'acrt', 'Training mode (fast | acrt)')
   cmd:option('-gdn', '', 'Global disparity network architecture')
   cmd:option('-mcnet', '', 'Path to MC trained network')
   cmd:option('-dispnet', '', 'Path to GDN trained network')
   cmd:option('-a', 'train_mcn | train_gdn | test | submit | time | predict', 'train_mc')
   cmd:option('-log', '../results', 'Logs dir')
   cmd:option('-gpu', 1, 'gpu id')
   cmd:option('-seed', 6, 'Random seed')
   cmd:option('-debug', false)
   cmd:option('-times', 1, 'Test the pipeline every X epochs')
   cmd:option('-after', 14, 'Test every epoch after this one')

   cmd:option('-all', false, 'Train on both train and validation sets')
   cmd:option('-rename', false, 'Rename the trained network')
   cmd:option('-mix', false, 'Train on both kitti and kitti15')
   cmd:option('-storage', '../storage', 'Path to dir with the training data')
   cmd:option('-name', '', 'Add string to the network name')
   cmd:option('-METHOD_NAME', 'ResMatch', 'Name for MB submission')
   cmd:option('-start_epoch', 1)
   cmd:option('-make_cache', false)
   cmd:option('-use_cache', false)
   cmd:option('-save_img', false, 'Save the images when testing')
   cmd:option('-sm_terminate', 'refinement', 'Terminate the stereo method after this step')
   cmd:option('-sm_skip', '', 'which part of the stereo method to skip')
   cmd:option('-subset', 1, 'Percentage of the data set used for training')
   cmd:option('-epochs', 15, 'The amount of epochs to train')
   cmd:option('-start_epoch', 1)
   cmd:option('-rect', 'imperfect')
   cmd:option('-color', 'rgb')
   cmd:option('-verbose', false)
   cmd:option('-inner', 'L', 'Inner skip-connection')
   cmd:option('-outer', 'L', 'Outer skip-connection')

	-- Parameters of the matching cost network
   cmd:option('-fm', 112)
   cmd:option('-nh2', 384)
   cmd:option('-margin', 0.2, '')
   cmd:option('-lambda', 0.8)
   cmd:option('-batch_size', 128)

   local opt = cmd:parse(arg)

   return opt

end

return M
