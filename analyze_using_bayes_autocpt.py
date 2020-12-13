from pandas import DataFrame
import pandas as pd
import math
import numpy as np
#from scipy.stats import variation

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from decimal import Decimal


fread=open("./data/final_report_UW1","r")
fread2=open("./data/map_standard","r")
d={}
for line in fread2.readlines():
	line=line.strip().split("\t")
	d[str(line[1])]=str(line[0])

d_df={'species':[],'hits':[],'region':[],'original_rank':[],'bayes_rank':[],
        '0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}

total_region=0
total_hits=0

with open ("./data/final_report_UW1","r") as f:
	for line in f.readlines():
		line=line.strip().split("\t")
		total_region=total_region+(len(line)-2)/2.0
		total_hits+=int(line[1])
	print(total_hits)
	print(total_region)
	ave_hits=math.ceil((total_hits*0.05)/total_region)
	print(ave_hits)

for line in fread.readlines():
	line=line.strip().split("\t")
	d_df['species'].append(d[str(line[0])])
	id_list=[str(k) for k in range(10)]
	flag=0
	sum_minus=0
	index_region=0
	for i in range(2,len(line),2):
		id=str(line[i])
		num=int(line[i+1])
		if num<=ave_hits:
			sum_minus+=num
			num=0
		else:
			index_region+=1
		id_list.remove(id)
		d_df[str(id)].append(num)
		flag+=1
	for j in id_list:
		d_df[j].append(0)
		flag+=1
	percent=(int(line[1])-sum_minus)/total_hits
	d_df['hits'].append(percent)
	d_df['region'].append(index_region)

hits_sort=sorted(d_df['hits'],reverse = True)
for i in d_df['hits']:
	loc=hits_sort.index(i)
	d_df['original_rank'].append(int(loc)+1)


## bayes
bayes_rank_values=[]

# clinic:['age','symptom','travel','sex','vaccination']
weights_all={'risk':[7,3]}

# basic functions for implementing the algorithm for auto-generating CPTs

def auto_p(num_of_states,index_of_states):
	p=Decimal(index_of_states)/Decimal(num_of_states-1.0)
	return p

def auto_score(weights_all,name,ps):
	sum_weight=0
	for i in weights_all[name]:
		sum_weight+=Decimal(i)
	sum=0
	for i in range(len(ps)):
		sum+=Decimal(weights_all[name][i]*(ps[i]))
	score=sum*Decimal(1.0)/(sum_weight)
	return score

def auto_area(score,num_of_child_state):
	area_list=[]
	prob_max_state=score
	prob_min_state=Decimal(1.0)-score
	#num_inner_points=num_of_child-1
	if prob_max_state>=prob_min_state:
		maxv=prob_max_state
		minv=prob_min_state
		diff=maxv-minv
		diff_ave=diff/Decimal(num_of_child_state)
		for i in range(num_of_child_state):
			area=(minv*Decimal(2)+diff_ave*(Decimal(2)*Decimal(i)+Decimal(1)))*(Decimal(1.0)/Decimal(num_of_child_state))
			area_list.append(area)
	else:
		maxv=prob_min_state
		minv=prob_max_state
		diff=maxv-minv
		diff_ave=diff*Decimal(1.0)/Decimal(num_of_child_state)
		for i in range(num_of_child_state):
			area=(maxv*Decimal(2)-diff_ave*(Decimal(2)*Decimal(i)+Decimal(1)))*(Decimal(1.0)/Decimal(num_of_child_state))
			area_list.append(area)
	return area_list

# define factors and thier status in the netwrok
def create_variables():
	hits={'name':'hits',
		'states':8
		}

	coverage={'name':'coverage',
		'states':10
		}

	risk={'name':'risk',
		'states':2
		}

	return hits,coverage,risk


def calculate_cpt_ranking_score(hits,coverage,risk):
	total_cpt_risk=[]
	for i in range(hits['states']):
		for j in range(coverage['states']):
				p_hits=auto_p(hits['states'],i)
				p_coverage=auto_p(coverage['states'],j)
				s=auto_score(weights_all,risk['name'], [p_hits,p_coverage])
				area_list=auto_area(s,risk['states'])
				total_cpt_risk.append(area_list)
	total_cpt_array=np.array(total_cpt_risk).T
	#print(total_cpt_array.shape)
	return total_cpt_array


## define the Bayesian network using pgmpy
def create_model(hits,coverage,risk):
	cpd_model = BayesianModel([('Hits','Risk'),('Coverage','Risk')])

	cpd_hits = TabularCPD(variable='Hits', variable_card=8,
                      values=[[0.125 for i in range(8)]])

	cpd_coverage = TabularCPD(variable='Coverage', variable_card=10,
                          values=[[0.1 for i in range(10)]])

	cpd_risk = TabularCPD(variable='Risk', variable_card=2,
                      values=calculate_cpt_ranking_score(hits,coverage,risk),
                      evidence=['Hits', 'Coverage'],
                      evidence_card=[8, 10]
                          )

	cpd_model.add_cpds(cpd_hits, cpd_coverage, cpd_risk)
	cpd_model.check_model()
	infer = VariableElimination(cpd_model)
	return infer


hits,coverage,risk=create_variables()
infer=create_model(hits,coverage,risk)


for i in range(len(d_df['hits'])):
	Evidence={}
	if d_df['hits'][i]<0.01:
		Evidence['Hits']=0
	elif 0.01 <= d_df['hits'][i] <0.03:
		Evidence['Hits']=1
	elif 0.03 <= d_df['hits'][i] <0.05:
		Evidence['Hits']=2
	elif 0.05 <= d_df['hits'][i] <0.1:
		Evidence['Hits']=3
	elif 0.1 <= d_df['hits'][i] <0.2:
		Evidence['Hits']=4
	elif 0.2 <= d_df['hits'][i] <0.3:
		Evidence['Hits']=5
	elif 0.3 <= d_df['hits'][i] <0.5:
		Evidence['Hits']=6
	else:
		Evidence['Hits']=7

	Evidence['Coverage']=d_df['region'][i]
	value = vars(infer.query(['Risk'], evidence=Evidence))['values'].tolist()
	max_value=value[1]
	bayes_rank_values.append(max_value)

hits_sort_bayes=sorted(bayes_rank_values,reverse = True)
for i in bayes_rank_values:
        loc=hits_sort_bayes.index(i)
        d_df['bayes_rank'].append(int(loc)+1)

df=DataFrame(d_df)
sorted=df.sort_values(by="hits",ascending= False)
print(sorted.head(20))
