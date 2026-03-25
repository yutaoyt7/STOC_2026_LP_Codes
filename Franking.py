
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

####### When this is set to true we addin the worse case analysis, which makes the solution a lowerbound for the approximation ratio. If this is set to False 
#######   the output becomes an upper bound, we can switch between True and False to get an estimation of the limit of LP as n goes to infinity
addworsecondition=True


###### This controls the parameter n, which splits the interval [0,1] into n uniform pieces
n = 5


###### first LP
###### Model settings, OptimalityTol is used to set the solution tolerance, when its set to 1e-k we are accurate up to 10^(-k)
model = gp.Model("GurobiLP")
model.setParam('OutputFlag', 1)
model.setParam('Method', 2)
model.setParam('OptimalityTol', 1e-5)
model.setParam("BarOrder", 1)
model.setParam("Presolve", 2)

#######         Variables defined
######  g:  the gain function with indicies from 0,1,...,n, we are using i to indicate interval [(i-1)/n, i/n), 0
######       0 is not used in the constriants, we have it simply because array starts indexing with 0
######  h:  the compensation function, 0 is used for indexing degenerate case
######  alpha: the approximation ratio, objective to maximize
######  active_gain:  active_gain[i] is a lower bound for u in [(i-1)/n, i/n) assuming u is matched actively or unmatched before introducing ustar 
######  passive_gain: passive_gain[i] is a lower bound for u in [(i-1)/n, i/n) assuming u is matched passively before introducing ustar
 
g = [model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'g[{i}]') for i in range(n+1)]
h = [model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'h[{i}]') for i in range(n+1)]
alpha = model.addVar(lb=0, ub=2, vtype=GRB.CONTINUOUS, name='alpha')
active_gain = [model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS, name=f'active_gain[{i}]') for i in range(n+1)]
passive_gain = [model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS, name=f'passive_gain[{i}]') for i in range(n+1)]

########### active_gain_vb_over_choice_of_theta:  lower bound to the active gain when u is matched actively before introducting ustar and has profile (u,v^A,b^A)
###########               the worst case is considered over adversarial choice of theta_0, theta_1
active_gain_vb_over_choice_of_theta = {
    (u, v, b): model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS,
                            name=f'active_gain_vb[{u},{v},{b}]')
    for u in range(n+1)
    for v in range(n+1)
    for b in range(v,n+1)  # ensures v < b
}

########### active_gain_vb_over_choice_of_theta:  lower bound to the active gain when u is matched actively before introducting ustar and has profile (u,v,no backup)
###########               the worst case is considered over adversarial choice of theta_0, theta_1
active_gain_v_over_choice_of_theta = {
    (u, v): model.addVar(lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS,
                         name=f'active_gain_v[{u},{v}]')
    for u in range(n+1)
    for v in range(n+1)
}

##########  short hand linear expression for the gain of vertex when vertex is matched actively with a victim. 
##########     active_with_victim[i] = 1-g[i]-h[i]
active_with_victim = []
for i in range(n+1):
    expr = gp.LinExpr()
    expr.addConstant(1)
    expr.addTerms(-1, g[i])
    expr.addTerms(-1, h[i])
    active_with_victim.append(expr)


##### Constarints for functions g,h
###### monotonicity property for g
for i in range(n):
    model.addConstr(g[i] <= g[i+1])   

###### monotonicity property for h, also enforcing the constraint that being matched is always better than unmatched while receiving 1 copy of compensation
###### Also making sure h[0]=0 
for i in range(1, n+1):
    if i>0:
        model.addConstr(active_with_victim[i] >=  h[n])
        model.addConstr(g[i]>= h[n])
    if i<n:
        model.addConstr(h[i] <= h[i+1])
model.addConstr(h[0]==0)




########## We now add constraints for each specific cases of us. To consider the worst possible cases, we make the following assumptions:
#########         1. b always lies on the left side of the interval, i.e. b=(i-1)/n + epsilon
#########                   (the only term that will change with b within interval [i-1/n,i/n) is of form (b-v)h(theta_0) , the worst case will happen when b is small)
#########         2. theta_0 always lies on the right side of the interval                
#########                    (The value changes with theta_0 within the interval takes form theta_0 * a + (theta_1-theta_0) * b where b>=a always, so the worst case happens when 
#########                          theta_0 = i/n-epsilon) 
#########         3. For theta_1 the worse case happens when theta_1 lies on the left side of the interval, but when theta_0 assumes the right most side of the interval and 
#########                   theta_0, theta_1 lying in the same interval, theta_1 is forces to take the right most position.
  



######  Constraint for u unmatched, profile (u, bot, bot)
for u in range(1,n+1):
    gain_no_match=gp.LinExpr()
    for i in range(1,n+1):
        gain_no_match.addTerms(1/n,g[i])
    model.addConstr(active_gain[u]<=gain_no_match)


######  Constraint for u matched passively with a passive backup at b,  profile (u, v^P, b^P)
for u in range(1,n+1):
    ###### constraint for u having a passive backup
    model.addConstr(passive_gain[u]<=g[u])    



###### constraits for u matched passively and has no backup, profile (u, v^P, bot)

for u in range(1,n+1):
    for theta_0 in range(0,n+1):
        ###### gain_us: the gain for the integrated parts of ustar
        gain_us=gp.LinExpr()
        for i in range(1,theta_0+1):
            gain_us.addTerms(1/n, g[i])
        model.addConstr(passive_gain[u]<=gain_us 
                                + max(n-theta_0,0)/n * h[theta_0]
                                + max(n-theta_0,0)/n * g[u]
                                )





######  Constraint for u matched passively with a active backup at b, profile (u, v^P, b^A)
for u in range(1,n+1):
    ###### constraints for u having an active backup at b
    for b in range(1,n+1):
        ####### us is short for ustar
        for theta_0 in range(0,n+1):
            ###### gain_us: the gain for the integrated parts of ustar
            gain_us=gp.LinExpr()
            for i in range(1,theta_0+1):
                gain_us.addTerms(1/n, g[i])
            model.addConstr(passive_gain[u]<=gain_us  
                                + max(b-theta_0-1,0)/n * h[theta_0]
                                + max(theta_0,0)/n * (active_with_victim[b])
                                + max(n-theta_0,0)/n * g[u])






##############       We consider theta_1 on the left side once for both u and ustar, we then consider theta_1 on the right for both u and ustar.
##############        Still assume worst case theta_0 on rightside of interval.


##########  Adding constraints when u has profile (u, v^A, bot).  
for v in range(1,n+1):
        ########## case when theta_1 lies on right side of interval, corresponds to case (1)(2) in paper.
        for theta_1 in range(v,n+1):
            gain_us=gp.LinExpr()
            for i in range(1,theta_1+1):
                gain_us.addTerms(1/n, g[i])
            for u in range(1,n+1):                
                for theta_0 in range(0, theta_1+1):
                    ######### case  1-f(v)-h(v) < g(u)
                    model.addConstr(active_gain_v_over_choice_of_theta[u,v]<=gain_us+ 
                                max(n-theta_1,0)/n * h[theta_0] +
                                max(theta_0,0)/n *h[v]+    
                                max(theta_1-theta_0,0)/n * active_with_victim[v]+
                                max(n-theta_1,0)/n * active_with_victim[v]
                                )

                    ######### case g(u) < 1-f(v)-h(v)            
                    model.addConstr(active_gain_v_over_choice_of_theta[u,v]<=gain_us+ 
                                max(n-theta_1,0)/n * h[theta_0] +
                                max(theta_0,0)/n * h[v]+   
                                max(theta_1-theta_0,0)/n * g[u] +
                                max(n-theta_1,0)/n * active_with_victim[v]
                                )
            if addworsecondition:
                ########## case when theta_1 lies on left side of interval,, corresponds to case (3)(4) in paper
                gain_us=gp.LinExpr()
                for i in range(1,theta_1):
                    gain_us.addTerms(1/n, g[i])
                for u in range(1,n+1):
                    for theta_0 in range(0, theta_1): 
                            ######### case  1-f(v)-h(v) < g(u)
                            model.addConstr(active_gain_v_over_choice_of_theta[u,v]<=gain_us+    
                                    max(n-theta_1+1,0)/n * h[theta_0] +
                                    max(theta_0,0)/n *h[v]+ 
                                    max(theta_1-theta_0-1,0)/n * active_with_victim[v] +
                                    max(n-theta_1+1,0)/n * active_with_victim[v]
                                    )
                            
                            ######### case g(u) < 1-f(v)-h(v)
                            model.addConstr(active_gain_v_over_choice_of_theta[u,v]<=gain_us+
                                    max(n-theta_1+1,0)/n * h[theta_0] +
                                    max(theta_0,0)/n *h[v]+   
                                    max(theta_1-theta_0-1,0)/n * g[u]  +
                                    max(n-theta_1+1,0)/n * active_with_victim[v]
                                    )






##############      This part bounds the expected gain when u has profile (u, v^A, b^A). This part adds the greatest number of constraints.  
##############      As usual, we assume b takes the left side of the interval and theta_0 takes the right side of the interval
##############    Adding Constraints
for b in range(1,n+1):
    for v in range(1,b+1):
        for theta_1 in range(v,n+1):
            ########## case when theta_1 lies on right side of interval, corresponds to case (1)(2)(3) in paper
            gain_us=gp.LinExpr()
            for i in range(1,theta_1+1):
                gain_us.addTerms(1/n, g[i])              
            for u in range(1,n+1):
                for theta_0 in range(0, theta_1+1):
                    ##########   case when 1-g(b)-h(b)< 1-g(v)-h(v) < g(u)
                    model.addConstr(active_gain_vb_over_choice_of_theta[u,v,b]<=gain_us+ 
                                max(b-theta_1-1,0)/n * h[theta_0] +
                                max(theta_0,0)/n * active_with_victim[b]+
                                max(theta_1-theta_0,0)/n * active_with_victim[v]+
                                max(n-theta_1,0)/n * active_with_victim[v]
                                )
                    
                    ##########  case  when 1-g(b)-h(b)< g(u) < 1-g(v)-h(v)           
                    model.addConstr(active_gain_vb_over_choice_of_theta[u,v,b]<=gain_us+ 
                                max(b-theta_1-1,0)/n * h[theta_0] +
                                max(theta_0,0)/n * active_with_victim[b]+
                                max(theta_1-theta_0,0)/n * g[u] +
                                max(n-theta_1,0)/n * active_with_victim[v]
                                )
                    
                    ##########  case  when g(u) < 1-g(b)-h(b) < 1-g(v)-h(v)            
                    model.addConstr(active_gain_vb_over_choice_of_theta[u,v,b]<=gain_us+ 
                                max(b-theta_1-1,0)/n * h[theta_0] +
                                max(theta_0,0)/n * g[u]+
                                max(theta_1-theta_0,0)/n * g[u] +
                                max(n-theta_1,0)/n * active_with_victim[v]
                                )
                    
            ########## case when theta_1 lies on left side of interval corresponds to case (4)(5)(6) in paper
            if addworsecondition:
                gain_us=gp.LinExpr()
                for i in range(1,theta_1):
                    gain_us.addTerms(1/n, g[i])  
                for u in range(1,n+1):
                    for theta_0 in range(0, theta_1):
                        ##########   case when 1-g(b)-h(b)< 1-g(v)-h(v) < g(u)
                        model.addConstr(active_gain_vb_over_choice_of_theta[u,v,b]<=gain_us+ 
                                    max(b-theta_1,0)/n * h[theta_0] +
                                    max(theta_0,0)/n * active_with_victim[b]+
                                    max(theta_1-theta_0-1,0)/n * active_with_victim[v]+
                                    max(n-theta_1+1,0)/n * active_with_victim[v]
                                    )
                        
                        ##########  case  when 1-g(b)-h(b)< g(u) < 1-g(v)-h(v)            
                        model.addConstr(active_gain_vb_over_choice_of_theta[u,v,b]<=gain_us+ 
                                    max(b-theta_1,0)/n * h[theta_0] +
                                    max(theta_0,0)/n * active_with_victim[b]+
                                    max(theta_1-theta_0-1,0)/n * g[u]+
                                    max(n-theta_1+1,0)/n * active_with_victim[v]
                                    )

                        ##########  case  when g(u) < 1-g(b)-h(b) < 1-g(v)-h(v)                                
                        model.addConstr(active_gain_vb_over_choice_of_theta[u,v,b]<=gain_us+ 
                                    max(b-theta_1,0)/n * h[theta_0] +
                                    max(theta_0,0)/n * g[u]+
                                    max(theta_1-theta_0-1,0)/n * g[u] +
                                    max(n-theta_1+1,0)/n * active_with_victim[v]
                                    )



########  These constraints aggregates the active gains at u by implementing the monotonicity properties
for u in range(1,n+1):
    ######### where the profile of u is (u,start_of_v, b], (u,start_of_v+1, b], ... (u,b,b] uniform distribution. 
    #########     we consider min(b+1,n) is because taking at the left sider of interval is the worst case, only except at ((n-1)/n,n/n], we cannot make b=n/n+epsilon
    for b in range(1,n+1):
        for start_of_v in range(1,b+1):
            active=gp.LinExpr()
            for v in range(start_of_v,b+1):
                active.addTerms( 1/(b+1-start_of_v), active_gain_vb_over_choice_of_theta[u,v,min(b+1,n)])
            model.addConstr(active_gain[u]<=active)

            active2=gp.LinExpr()
            for v in range(start_of_v,b+1):
                active2.addTerms( 1/(b+1-start_of_v), active_gain_vb_over_choice_of_theta[u,v,b])
            model.addConstr(active_gain[u]<=active2)

    ###########     where the profile of u is (u,start_of_v, no backup), (u,start_of_v+1, no backup), ... (u,n/n, no_backup) uniform distribution. 
    for start_of_v in range(1,n+1):
        active=gp.LinExpr()
        for v in range(start_of_v,n+1):
            active.addTerms( 1/(n+1-start_of_v), active_gain_v_over_choice_of_theta[u,v])
        model.addConstr(active_gain[u]<=active)


############# adding constraints by considering randomization over u
#############     theta_u is the position u transitions from being passively matched to activly finding a match
for theta_u in range(n+1):
    passive=gp.LinExpr()
    active=gp.LinExpr()
    for i in range(1,theta_u+1):
        passive.addTerms(1/n,passive_gain[i])
    for i in range(theta_u+1,n+1):
        active.addTerms(1/n,active_gain[i])
    model.addConstr(alpha<=active+passive)






# Objective
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


# if model.status == GRB.OPTIMAL:
#     # Print f values
#     print("Solution for f:")
#     for i, var in enumerate(g):
#         print(f"{var.VarName} = {var.X}")

#     # Print h values
#     print("\nSolution for h:")
#     for i, var in enumerate(h):
#         print(f"{var.VarName} = {var.X}")