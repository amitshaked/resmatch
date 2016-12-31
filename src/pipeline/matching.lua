
local M = {}

function M.match(mcnet, x_batch, disp_max, directions)
		return mcnet:computeMatchingCost(x_batch, disp_max, directions)
end

return M
