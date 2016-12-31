local M = {}

local Logger = torch.class('Logger', M)

function Logger:__init(opt)
   local cmd = getCmd(opt)
   self.path = opt.log .. '/' .. cmd .. '.txt'
   local writer = io.open(self.path, 'a')
   writer:write(os.date([[%x %X]])..'\n')
   writer:close()
end

function Logger:write(str)
   print(str)
   local writer = io.open(self.path, 'a')
   writer:write(str)
   writer:close()
end

function getCmd(opt)

   local cmd_str = opt.a
   local i =1
   while i <= #arg do
      if arg[i] == '-mcnet' or arg[i] == '-dispnet' or arg[i] == '-sm_skip' or arg[i] == '-sm_terminate' then
         i = i +2
      elseif arg[i] == '-make_cache' or arg[i] == '-use_cache' or arg[i] == '-save_train' then
         i = i +1
      else
         cmd_str = cmd_str .. '_' .. arg[i]
         i = i+1
      end
   end

   return cmd_str
end

return M.Logger
