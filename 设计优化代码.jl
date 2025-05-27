using JuMP
using Ipopt
using Gurobi
using BlackBoxOptim
using BlackBoxOptim: num_func_evals
using Plots
using CSV, DataFrames

function energy(vs::Vector) #风力机模型
	η_t = 0.96
	η_g = 0.93
	h1 = 10.0
	h2 = 135.0
	α = 1.0 / 7.0
	Δt = 1.0
	η_inverter = 0.98
	return map(copy(vs) * (h2 / h1)^α) do v
			   if v < 3.0
				   return 0.0
			   elseif v < 9.5
				   return (-30.639 * v^3 + 623.5 * v^2 - 3130.4 * v + 4928) / 5000.0
			   elseif v < 19.5
				   return 1.0
			   elseif v < 25.0
				   return (-203.97 * v + 9050.9) / 5000.0
			   else
				   return 0.0
			   end
		   end * η_t * η_g * Δt * η_inverter
end

function energy(G::Vector, T::Vector) #太阳能电池模型
	return 0.9 * (1.0 .- 0.004 * (T .- 25.0)) .* G / 1000
end

function generate_load()
	plan = [0.84, 0.78, 0.78, 0.74, 0.83, 0.94, 0.78]
	days = [
		[0.4, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.7, 0.4],
		[0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.8, 0.7, 0.6, 0.6, 0.6, 0.7, 0.5],
		[0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.6, 0.6],
		[0.4, 0.4, 0.4, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0, 0.8, 0.8, 0.7, 0.7, 0.9, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.4, 0.4],
		[0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6],
		[0.9, 0.8, 0.9, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9],
		[0.6, 0.7, 0.6, 0.4, 0.4, 0.4, 0.4, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.7],
	]
	load_Si = 12.9 * vcat((plan[i] .* days[i] for i in 1:7)...)
	electricity_load = (31700 + 600) * load_Si
	hydrogen_load = 175 * 3.25 * load_Si # 1Nm³氢气相当于3.25kWh电
	heat_load = (4.2 * 751.13 + 2.9 * 740.2 + 1 * 735.7) * load_Si
	return (electricity_load, hydrogen_load, heat_load) # kW 
end

function generate_electricity_price()
	price = [3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 3, 3, 2, 1, 1, 2, 3, 3, 4, 4, 5, 5, 4]
	price_mapping = Dict(1 => 0.1049, 2 => 0.1411, 3 => 0.3223, 4 => 0.5035, 5 => 0.5639)
	price_grid = [get(price_mapping, p, p) for p in price]
	return vcat([price_grid for _ in 1:9]...)
end

const T = 1:24*9  #优化周期
const M=1e6
const factor_wt = CSV.read(joinpath(@__DIR__, "output.csv"), DataFrame)[:, 4][T]
const factor_pv = CSV.read(joinpath(@__DIR__, "output.csv"), DataFrame)[:, 3][T]
const factor_load= CSV.read(joinpath(@__DIR__, "output.csv"), DataFrame)[:, 5][T] #读取代表日
function simulate(x::Vector{Float64})
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    price = generate_electricity_price()
    print(length(price))
    price_Q=price/4
    H_Load=175 * 3.25*factor_load
    Q_Load=(4.2 * 751.13 + 2.9 * 740.2 + 1 * 735.7) *factor_load
    L_Load=(31700 + 600) *factor_load
    @variable(model,priceH[t in T] >=0)
    @constraint(model,[t in T],priceH[t]<=1)
    @variable(model,λ[t in T]>=0)
    @variable(model,μ1[t in T]>=0)
    @variable(model,μ2[t in T]>=0)
    @variable(model,H_buy[t in T]>=0)
    @variable(model,H_ET2[t in T]>=0)
    @variable(model,E_ET2[t in T]>=0)
    @variable(model,μ3[t in T]>=0)
    IC_ET2=5e4
    @variable(model,λ2[t in T]>=0)
    @variable(model,μ4[t in T]>=0)
    @variable(model,μ5[t in T]>=0)
    @variable(model,Q_buy[t in T]>=0)
    @variable(model,Q_HP[t in T]>=0)
    @variable(model,E_HP[t in T]>=0)
    @variable(model,μ6[t in T]>=0)
    @variable(model,priceQ[t in T] >=0)
    @constraint(model,[t in T],priceQ[t]<=1)
    IC_HP=2e4
    IC_ET = x[1]
    @variable(model, E_ET[t in T] >= 0)
    @variable(model, H_ET[t in T] >= 0)
    @constraint(model, [t in T], H_ET[t] == E_ET[t] * 0.63)
    @constraint(model, [t in T], E_ET[t] <= IC_ET)
    @variable(model, H_ET_SELL[t in T] >= 0)
    @constraint(model, [t in T], H_ET_SELL[t] == H_buy[t])

    IC_FC = x[2]
    @variable(model, H_FC[t in T] >= 0)
    @variable(model, E_FC[t in T] >= 0)
    @variable(model, Q_FC[t in T] >= 0)
    @variable(model, H_FC2[t in T] >= 0)
    @variable(model, E_FC2[t in T] >= 0)
    @variable(model, Q_FC2[t in T] >= 0)
    @constraint(model, [t in T], E_FC[t] == H_FC[t] * 0.57)
    @constraint(model, [t in T], Q_FC[t] == H_FC[t]*0.28)
    @constraint(model, [t in T], E_FC2[t] == H_FC2[t] * 0.57)
    @constraint(model, [t in T], Q_FC2[t] == H_FC2[t]*0.28)
    @constraint(model, [t in T], Q_FC2[t] <= Q_buy[t])
    @constraint(model, [t in T], E_FC2[t]+E_FC[t] <= IC_FC*0.57)

    IC_ES = x[3]
    @variable(model, E_ES[t in T] >= 0)
    @variable(model, E_ES_in[t in T] >= 0)
    @variable(model, E_ES_out[t in T] >= 0)
    ηI_ES = 0.96
    ηO_ES = 0.96
    r_ES = 0.5
    min_ES = 0.1
    max_ES = 1.0
    ini_ES = 0.2
    loss_ES = 0.01
    @constraint(model, E_ES[T[1]] == ini_ES * IC_ES * (1 - loss_ES) + E_ES_in[T[1]] * ηI_ES - E_ES_out[T[1]] / ηO_ES)
    @constraint(model, [t in T[2:end]], E_ES[t] == E_ES[t-1] * (1 - loss_ES) + E_ES_in[t] * ηI_ES - E_ES_out[t] / ηO_ES)
    @constraint(model, [t in T], E_ES_in[t] <= r_ES * IC_ES)
    @constraint(model, [t in T], E_ES_out[t] <= r_ES * IC_ES)
    @constraint(model, [t in T], min_ES * IC_ES <= E_ES[t])
    @constraint(model, [t in T], E_ES[t] <= max_ES * IC_ES)
    @constraint(model, E_ES[T[end]] == ini_ES * IC_ES)
    @variable(model,γ[t in T],Bin)
    @constraint(model,[t in T], E_ES_in[t]<=γ[t]*M)
    @constraint(model,[t in T],E_ES_out[t]<=(1-γ[t])*M)

    IC_HS = x[4]
    @variable(model, H_HS[t in T] >= 0)
    @variable(model, H_HS_in[t in T] >= 0)
    @variable(model, H_HS_out[t in T] >= 0)
    ηI_HS = 0.8
    ηO_HS = 0.8
    r_HS = 0.25
    min_HS = 0.1
    max_HS = 1.0
    ini_HS = 0.15
    loss_HS = 0.001
    @constraint(model, H_HS[T[1]] == ini_HS * IC_HS * (1 - loss_HS) + H_HS_in[T[1]] * ηI_HS - H_HS_out[T[1]] / ηO_HS)
    @constraint(model, [t in T[2:end]], H_HS[t] == H_HS[t-1] * (1 - loss_HS) + H_HS_in[t] * ηI_HS - H_HS_out[t] / ηO_HS)
    @constraint(model, [t in T], H_HS_in[t] <= r_HS * IC_HS)
    @constraint(model, [t in T], H_HS_out[t] <= r_HS * IC_HS)
    @constraint(model, [t in T], min_HS * IC_HS <= H_HS[t])
    @constraint(model, [t in T], H_HS[t] <= max_HS * IC_HS)
    @constraint(model, H_HS[T[end]] == ini_HS * IC_HS)

    IC_WT = x[5]
    P_WT = IC_WT * factor_wt
    @variable(model, E_WT[t in T] >= 0)
    @variable(model,E_WT_dis[t in T]>=0)
    @constraint(model, [t in T], E_WT[t] <= P_WT[t])
    @constraint(model, [t in T], E_WT[t]+E_WT_dis[t]==P_WT[t])

    IC_PV = x[6]
    P_PV = IC_PV * factor_pv
    @variable(model, E_PV[t in T] >= 0)
    @variable(model, E_PV_dis[t in T]>=0)
    @constraint(model, [t in T], E_PV[t] <= P_PV[t])
    @constraint(model, [t in T], E_PV[t]+E_PV_dis[t]==P_PV[t])

    @variable(model, E_BUY[t in T] >= 0)
    @variable(model, E_SELL[t in T] >= 0)
    @constraint(model,[t in T],E_SELL[t] <= (L_Load[t]+E_ET2[t]+E_HP[t])*0.2)

    @constraint(model, [t in T], E_WT[t] + E_PV[t] + E_BUY[t] + E_FC[t] + E_FC2[t] + E_ES_out[t] == E_ET[t] + E_ES_in[t] + E_SELL[t])
    @constraint(model, [t in T], H_ET[t] + H_HS_out[t] == H_FC[t] +H_FC2[t]+ H_HS_in[t]+H_ET_SELL[t])
    @constraint(model,[t in T],price[t]-λ[t]*(0.71)-μ1[t]+μ3[t]==0)
    @constraint(model,[t in T],priceH[t]-λ[t]-μ2[t]==0)
    @variable(model,γ1[t in T],Bin)
    @constraint(model,[t in T],μ1[t]<=(1-γ1[t])*M)
    @constraint(model,[t in T],E_ET2[t]<= γ1[t]*M)
    @variable(model,γ2[t in T],Bin)
    @constraint(model,[t in T],μ2[t]<=(1-γ2[t])*M)
    @constraint(model,[t in T],H_buy[t]<= γ2[t]*M)
    @constraint(model,[t in T],H_ET2[t]==0.71*E_ET2[t])
    @constraint(model,[t in T],H_buy[t]+H_ET2[t]==H_Load[t])
    @variable(model,γ3[t in T],Bin)
    @constraint(model,[t in T],μ3[t]<=(1-γ3[t])*M)
    @constraint(model,[t in T],IC_ET2-E_ET2[t]<= γ3[t]*M)
    @constraint(model,[t in T],price[t]-λ2[t]*4-μ4[t]+μ6[t]==0)
    @constraint(model,[t in T],priceQ[t]-λ2[t]-μ5[t]==0)
    @variable(model,γ4[t in T],Bin)
    @constraint(model,[t in T],μ4[t]<=(1-γ4[t])*M)
    @constraint(model,[t in T],E_HP[t]<= γ4[t]*M)
    @variable(model,γ5[t in T],Bin)
    @constraint(model,[t in T],μ5[t]<=(1-γ5[t])*M)
    @constraint(model,[t in T],Q_buy[t]<= γ5[t]*M)
    @constraint(model,[t in T],Q_HP[t]==4*E_HP[t])
    @constraint(model,[t in T],Q_buy[t]+Q_HP[t]==Q_Load[t])
    @variable(model,γ6[t in T],Bin)
    @constraint(model,[t in T],μ6[t]<=(1-γ6[t])*M)
    @constraint(model,[t in T],IC_HP-E_HP[t]<= γ6[t]*M)
    @objective(model, Max, (sum(Q_FC2.*price_Q)+sum(E_SELL .* price*0.95)+sum(H_ET_SELL.*priceH))-sum(price .* (E_BUY)))
    set_optimizer_attribute(model, "OutputFlag", 1)   
    set_optimizer_attribute(model, "TimeLimit", 50)  # 设置时间限制为100秒
    set_optimizer_attribute(model, "MIPGap", 0.01)  #设置最优条件，在1%条件下求解时间可接受 
    optimize!(model)
	if termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        inv1=IC_ET * 6000 +  IC_FC * 8000 + IC_ES * 2000 + IC_HS * 4000 + IC_WT * 4300 + IC_PV * 3500
        reward=objective_value(model)/9
        cOM2=IC_ET*26+IC_FC*26+IC_ES*26+IC_HS*26+IC_PV*26+IC_PV*14
        depreciation=(IC_ET * 6000*2 +  IC_FC * 8000 + IC_ES * 2000*2 + IC_HS * 4000 + IC_WT * 4300 + IC_PV * 3500)/20
        inv2=0
        for i in 0:9
            inv=(IC_ET * 6000 +IC_ES * 2000)*(10-i)/10 +  (IC_FC * 8000 +  IC_HS * 4000 + IC_WT * 4300 + IC_PV * 3500)*(19-i)/20
            inv2=inv2+(reward*365-cOM2-inv*0.25/100-(reward*365-depreciation)*0.25)/(1.06^(i+1))
        end
        inv=(IC_ET * 6000 +IC_ES * 2000)*(19-10)/10 +  (IC_FC * 8000 +  IC_HS * 4000 + IC_WT * 4300 + IC_PV * 3500)*(19-10)/20
        inv2=inv2+(reward*365-(IC_ET * 6000 +IC_ES * 2000)-cOM2-inv*0.25/100-(reward*365-depreciation)*0.25)/(1.06^(10+1))
        for i in 11:19
            inv=(IC_ET * 6000 +IC_ES * 2000)*(19-i)/10 +  (IC_FC * 8000 +  IC_HS * 4000 + IC_WT * 4300 + IC_PV * 3500)*(19-i)/20
            inv2=inv2+(reward*365-cOM2-inv*0.25/100-(reward*365-depreciation)*0.25)/(1.06^(i+1))
        end
		return inv1-inv2 #计算NPV
	else
		return 5e19
	end
end

history = Array{Tuple{Int, Float64}, 1}()
callback = oc -> push!(history, (num_func_evals(oc), best_fitness(oc)))

res = bboptimize(
	simulate;
	SearchRange = [(0.5e4, 3e5), (3e3, 5e5), (0.7e4, 1.55e5), (0.7e4, 1.55e5), (0.7e3, 1.1e5), (0.7e3, 1.1e5)],
	Method = :adaptive_de_rand_1_bin_radiuslimited,
	MaxTime = 200,
	NumDimensions = 6,
	TraceMode = :compact,
	SaveTrace = true,
	PopulationSize = 100,
	CallbackFunction = callback, CallbackInterval = 0.0,
)

plot(history)
df = DataFrame(
    id = [x[1] for x in history],      # 提取每个元组的第一个元素（Int）
    value = [x[2] for x in history]    # 提取每个元组的第二个元素（Float64）
)

CSV.write("history.csv", df)