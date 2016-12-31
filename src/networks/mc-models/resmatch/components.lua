
local resmatch = {}

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride, shortcutType)
   if shortcutType == 'A' or shortcutType == 'B' or shortcutType == 'C' then
      local useConv = shortcutType == 'C' or
      (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
         :add(Convolution(nInputPlane, nOutputPlane, 1, 1))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
         :add(nn.SpatialAveragePooling(1, 1, stride, stride))
         :add(nn.Concat(2)
         :add(nn.Identity())
         :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   elseif shortcutType == 'D' then
      local m = nn.Mul()
      m.weight:fill(1)
      return nn.Sequential()
      :add(Convolution(nInputPlane, nOutputPlane, 1, 1))
      :add(m)
   elseif shortcutType == 'L' then
      local m = nn.Mul()
      m.weight:fill(1)
      return m
   end
end

local function residualBlock(model, block, fin, fout, shortcutType)

   concat = nn.ConcatTable()

   concat:add(block(fin, fout, stride))
   concat:add(shortcut(fin, fout, 1, shortcutType))
   model:add(concat):add(nn.CAddTable())
end

function basicBlock(fin, fout, stride)
   block = nn.Sequential()
   block:add(Convolution(fin,fout,3,3,stride,stride,1,1))

   block:add(Activation())
   block:add(Convolution(fout,fout,3,3,1,1,1,1))

   return block
end

function resmatch.transition(fin, fout)
   local stack = nn.Sequential()
   -- Convolution
   stack:add(Convolution(fin, fout, 3, 3))

   --Activation
   stack:add(Activation())
   return stack
end

local function innerResStack(block, f, n, stride, shortcut, last)

   local stack = nn.Sequential()
   for k=1, n do
      residualBlock(stack, block, f, f, shortcut)
      if k < n or not last then
         --stack:add(Activation()) -- better results with no activation?
      end
   end
   return stack
end

function resmatch.resStack(block, f, nOut, nIn, stride, shortcutIn, shortcutOut, last)

   local stack = nn.Sequential()

   for i=1, nOut do
      local innerBlock = innerResStack(block, f, nIn, 1, shortcutIn, i == nOut and last)
      if shortcutOut and shortcutOut ~= 'none' and nIn > 1 then
         stack:add(nn.ConcatTable()
            :add(innerBlock)
            :add(shortcut(f,f,1,shortcutOut)))
         :add(nn.CAddTable())
      else
         stack:add(innerBlock)
      end
      if not last then
         --stack:add(Activation())
      end
   end

   return stack
end

return resmatch
