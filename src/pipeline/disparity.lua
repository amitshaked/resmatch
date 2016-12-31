
local M = {}

function M.disparityImage(costs, gdn)
	if gdn then
		return gdn:disparityImage(costs)
	else
		local v1, d1 = torch.min(costs[{{1}}], 2)
		local disp = {}
		disp[2] = d1:cuda():add(-1)

		local v2 = torch.CudaTensor()
		local d2 = torch.CudaTensor()
		if costs:size(1) > 1 then
			v2, d2 = torch.min(costs[{{2}}], 2)
			disp[1] = d2:cuda():add(-1)
		end
		return disp, costs, {v1, v2}, {t1 = 1000, t2 = 1000}
	end
end

return M
