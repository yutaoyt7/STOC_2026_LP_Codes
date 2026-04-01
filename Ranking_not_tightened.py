######### These are imported Libraries
import gurobipy as gp
from gurobipy import GRB, quicksum 
import time
import gc
import os
import psutil
import sys


########## This is used to calculate how much memory and running time are used for the program to get an estimation of how scalable the LP is
process = psutil.Process(os.getpid())
print(f"Memory usage before model build: {process.memory_info().rss / (1024**2):.2f} MB")
memorybf=process.memory_info().rss / (1024**2)


total_start=time.time()
start = time.time()


###### Various parameters
######## When this is set to true the program outputs total running time
printtime=True


###### This controls the parameter n, which splits the interval (0,1] into n uniform pieces
# Read n from command line
# if len(sys.argv) < 2:
#     print("Usage: python Ranking.py <n>")
#     sys.exit(1)


n = 10



###### first LP
###### Model settings, OptimalityTol is used to set the solution tolerance, when its set to 1e-k we are accurate up to 10^(-k)
model = gp.Model("GurobiLP")
model.setParam("Method", 2)          # Barrier method (best for large continuous LPs)
model.setParam("Crossover", 0)       # Skip crossover to save time (if you only need the LP optimum)
model.setParam("Threads", 64)        # Match to physical cores on one socket
model.setParam("Presolve", 1)        # Aggressive presolve
model.setParam("BarOrder", 1)        # Use approximate minimum degree ordering (usually best)
model.setParam("BarConvTol", 1e-6)   # Tight barrier convergence tolerance
model.setParam("OptimalityTol", 1e-4) # Default or slightly tighter if needed
model.setParam("OutputFlag", 1)      # Keep log output visible


#######         Variables defined
######  g:  the gain function with indicies from 0,1,...,n, we assume that u (buyer) matched actively to v(product) with u in range [(i-1)/n, i/n), v in range [(j-1)/n, j/n). 
######         u gets 1-g(i,j) and v gets g(i,j) respectively
######  h:  the compensation function
######  alpha: the approximation ratio, objective to maximize
######  gain:  gain[i] is a lower bound for the gain of u + gain of ustar for u in [(i-1)/n, i/n)                                       (G_u function)
######  gain_no_match: gain_no_match[i] is a lower bound for u in [(i-1)/n, i/n) assuming u is unmatched before ustar gets added        (G function for profiles (x_u, \bot, \bot))
######  gain_no_backup: gain_no_backup[i,j] is a lower bound for u in [(i-1)/n, i/n), v in [(j-1)/n, j/n)                               (G function for profiles (x_u, x_v, \bot))
######                 assuming u is matched to v and does not have a backup before ustar gets added
######  gain_backup: gain_backup[i,j,k] is a lower bound for u in [(i-1)/n, i/n), v in [(j-1)/n, j/n), b in [(k-1)/n, k/n)              (G function for profiles (x_u, x_v, x_b))
########             assuming u is matched to v and has a backup b before ustar gets added
g = {
    (u, v): model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                         name=f'g[{u},{v}]')
    for u in range(n+1)
    for v in range(n+1)
}

h = {
    (u, v): model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                         name=f'h[{u},{v}]')
    for u in range(n+1)
    for v in range(n+1)
}

alpha = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='alpha')

gain = [model.addVar(lb=0, ub=2, vtype=GRB.CONTINUOUS, name=f'gain[{i}]') for i in range(n+1)]

gain_no_match = {
    (u): model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS,
                         name=f'gain_no_match [{u}]')
    for u in range(n+1)
}


gain_no_backup= {
    (u, v): model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS,
                         name=f'active_gain_v[{u},{v}]')
    for u in range(n+1)
    for v in range(n+1)
}

gain_backup= {
    (u, v, b): model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS,
                         name=f'active_gain_v[{u},{v},{b}]')
    for u in range(n+1)
    for v in range(n+1)
    for b in range(v,n+1)
}

################# In most cases, we omit the i in indexing and use the subscripts directly instead. For example, v is used instead of i_v.

##########  short hand linear expression for the gain of vertex u (buyer) matched to v (product) and each has a victim
##########     buyer_with_victim[u,v] = 1-g[u,v]-h[u,v]   
##########     product_with_victim[v1,v2] = g[u,v]-h[v,u]  
buyer_with_victim={}
product_with_victim={}

for u in range(1,n+1):
    for v in range(1,n+1):
        gain_buyer_with_victim=gp.LinExpr()
        gain_buyer_with_victim.addConstant(1)
        gain_buyer_with_victim.addTerms(-1,g[u,v])
        gain_buyer_with_victim.addTerms(-1,h[u,v])
        buyer_with_victim[u,v]=gain_buyer_with_victim


        gain_product_with_victim=gp.LinExpr()
        gain_product_with_victim.addTerms(1,g[u,v])
        gain_product_with_victim.addTerms(-1,h[v,u])
        product_with_victim[u,v]=gain_product_with_victim


##### function constraints (Def 5.8)
##### Add monotonicity constraints for g,h and adding constraints that enforces unmatched with <=4 compensation always worse than matched     
for u in range(1,n+1):
    for v in range(1,n+1):

        #############.  monotonicity constraints for g 
        if v<n:
            model.addConstr(g[u,v] <= g[u,v+1])
        if u<n:
            model.addConstr(g[u,v] >= g[u+1,v])

        ###########       Constraints for being matched better than receiving compensations. 
        model.addConstr(buyer_with_victim[u,v]>=4*h[1,n])
        model.addConstr(product_with_victim[u,v]>=4*h[1,n])

#############.  monotonicity constraints for h, also setting h(x,0)=0 
for u in range(0,n+1):
    model.addConstr(h[u,0] == 0)
    for v in range(0,n+1):
        if v<n:
            model.addConstr(h[u,v] <= h[u,v+1])
        if u<n:
            model.addConstr(h[u,v] >= h[u+1,v])



########## constraints for gain_no_match, the case where u is unmatched before adding ustar         (Claim 5.18)
for u in range(1,n+1):
    exp_gain=gp.LinExpr()
    for us in range(1,n+1):
        exp_gain+=1/n * product_with_victim[u,us]
    model.addConstr(gain_no_match[u]<= exp_gain)


######### constraints for gain_no_backup, the case where u is matched to v and has no backup b.      (Claim 5.24)
for u in range(1,n+1):
    for v in range(1,n+1):
        ###############       (Constraint family (1), case i_theta_0 <= i_u <= i_v)
        if u<=v:
            for theta_0 in range(0,u+1):
                
                cumulative_of_us=gp.LinExpr()

                for us in range(1,v):
                    cumulative_of_us += 1/n * product_with_victim[u,us]
                cumulative_of_us += 1/n * 1/2  * product_with_victim[u,v]

                for us in range(1,theta_0+1):
                    cumulative_of_us += 1/n * h[us,u]
                
                model.addConstr(
                    gain_no_backup[u,v]<=cumulative_of_us+
                    max(n-v,0)/n * h[u,theta_0]+
                    max(n-v,0)/n * h[v,u]+
                    theta_0/n * h[u,v]+
                    max(n-theta_0,0)/n * buyer_with_victim[u,v]
                )

        if u>=v:
            for theta_0 in range(0,u+1):
                ###############       (Constraint family (2), case i_theta_0, i_v <= i_u )
                cumulative_of_us=gp.LinExpr()
                for us in range(1,theta_0+1):
                    cumulative_of_us += 1/n * g[v,us]
                for us in range(theta_0+1,v):
                    cumulative_of_us+=1/n * product_with_victim[u,us]
                model.addConstr(
                    gain_no_backup[u,v]<=cumulative_of_us+
                    max(n-max(theta_0,v-1),0)/n * h[v,theta_0]+
                    max(n-max(theta_0,v-1),0)/n * h[v,u]+
                    theta_0/n * h[u,v]+
                    max(n-theta_0,0)/n *  buyer_with_victim[u,v]
                )

                ###########     (Constraint family (3), case    i_v <= i_u while i_u-1 <= i_theta_0 <= i_u )
                if  theta_0>=u-1: 
                    cumulative_of_us=gp.LinExpr()
                    for us in range(1,theta_0+1):
                        cumulative_of_us += 1/n * g[v,us]
                    model.addConstr(
                        gain_no_backup[u,v]<=cumulative_of_us+
                        max(n-theta_0,0)/n * h[v,u]+
                        (n-theta_0)/n * buyer_with_victim[u,v]
                    )




############ constraints for gain_backup, the case where u is matched to v and has a backup b.  (Claim 5.27)
for u in range(1,n+1):
    for v in range(1,n+1):
        for b in range(v,n+1):
            if u<=v:
                for theta_0 in range(0,u+1):
                    if v<b:
                         ###############       (Constraint family (1.1), case i_theta_0 <= i_u <= i_v < i_b)
                        cumulative_of_us=gp.LinExpr()
                        for us in range(1,v):
                            cumulative_of_us+= 1/n * product_with_victim[u,us]
                        cumulative_of_us += 1/n *1/2 *product_with_victim[u,v]
                        model.addConstr(
                            gain_backup[u,v,b]<=cumulative_of_us+
                            max(b-v-1,0)/n * h[u,theta_0]+
                            max(b-v-1,0)/n * h[v,u]+
                            theta_0/n * buyer_with_victim[u,b]+
                            (n-theta_0)/n * buyer_with_victim[u,v]
                        )
                    else:
                    ###############       (Constraint family (1.2), case i_theta_0 <= i_u <= i_v = i_b)
                        cumulative_of_us=gp.LinExpr()
                        for us in range(1,v):
                            cumulative_of_us+= 1/n * product_with_victim[u,us]

                        model.addConstr(
                            gain_backup[u,v,b]<=cumulative_of_us+
                            theta_0/n * buyer_with_victim[u,b]+
                            (n-theta_0)/n * buyer_with_victim[u,v]
                        )
            if u>=v:
                for theta_0 in range(0,u+1):
                    ###############       (Constraint family (2), case i_theta_0, i_v <= i_u )
                    cumulative_of_us=gp.LinExpr()
                    for us in range(1,theta_0+1):
                        cumulative_of_us+=1/n *product_with_victim[v,us]
                    for us in range(theta_0+1,v):
                        cumulative_of_us+=1/n * product_with_victim[u,us]
                    model.addConstr(
                        gain_backup[u,v,b]<=cumulative_of_us+
                        max(b-1-max(theta_0,v-1),0)/n * h[v,theta_0]+
                        max(b-1-max(theta_0,v-1),0)/n * h[v,u]+
                        theta_0/n * buyer_with_victim[u,b]+
                        (n-theta_0)/n * buyer_with_victim[u,v]
                    )
                    ###########     (Constraint family (3), case    i_v <= i_u while i_u-1 <= i_theta_0 <= i_u )
                    if theta_0>= u-1:
                        cumulative_of_us=gp.LinExpr()
                        for us in range(1,theta_0+1):
                            cumulative_of_us+=1/n *product_with_victim[v,us]
                        model.addConstr(
                            gain_backup[u,v,b]<=cumulative_of_us+
                            max(b-theta_0-1,0)/n * h[v,u]+
                            theta_0/n * buyer_with_victim[u,b]+
                            (n-theta_0)/n * buyer_with_victim[u,v]
                        )


#########     Aggregating the profile constraints (Claim 5.28, Claim 7.2)
for u in range(1,n+1):
    #########  constraint for case not matched (x_u,\bot,\bot)
    model.addConstr(gain[u]<= gain_no_match[u])

    #########  constraint for case no backup (x_u,x_v,\bot)
    for start_of_v in range(1,n+1):
        exp_gain_nobackup=gp.LinExpr()
        for v in range(start_of_v,n+1):
            exp_gain_nobackup+=1/(n+1-start_of_v) * gain_no_backup[u,v]
        model.addConstr(gain[u]<= exp_gain_nobackup)


    #########  constraint for case with backup (x_u,x_v,x_b)   
    for start_of_v in range(1,n+1):
        for b in range(start_of_v,n+1):
            exp_gain_backup=gp.LinExpr()
            for v in range(start_of_v,b+1):
                exp_gain_backup+=1/(b+1-start_of_v) * gain_backup[u,v,b]
            model.addConstr(gain[u]<= exp_gain_backup)

            exp_gain_backup2=gp.LinExpr()
            for v in range(start_of_v,b+1):
                exp_gain_backup2+=1/(b+1-start_of_v) * gain_backup[u,v,min(b+1,n)]
            model.addConstr(gain[u]<= exp_gain_backup2)





###### Integrating over worst case profile distributions (Claim 7.4)
exp_gain_overall=gp.LinExpr()
for u in range(1,n+1):
    exp_gain_overall+=1/n * gain[u]
model.addConstr(alpha<= exp_gain_overall)



    


if printtime:
    print("Wall-clock time setting constraints:", time.time() - start)
start=time.time()
# Set objective
model.setObjective(alpha, GRB.MAXIMIZE)

print(f"Memory usage for first model: {process.memory_info().rss / (1024**2) - memorybf:.2f} MB")


# Solve
model.optimize()

# Output
if model.status == GRB.OPTIMAL:
    print('Optimal solution found.')
    print('Objective value:', model.objVal)
elif model.status == GRB.FEASIBLE:
    print('Suboptimal feasible solution found.')
else:
    print('No feasible solution found.')

if printtime:
    print("Wall-clock time solving first LP:", time.time() - start)
start=time.time()


# for u in range(n+1):
#     for v in range(n+1):
#         print(f"g[{u},{v}] = {g[u,v].X}")

## Print h values
# print("\nSolution for h:")
# for u in range(n+1):
#     for v in range(n+1):
#         print(f"h[{u},{v}] = {h[u,v].X}")