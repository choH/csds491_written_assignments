from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianModel([("Battery", "Gauge"),("Battery", "Turns Over"),("Turns Over", "Starts"), ("Fuel", "Gauge"),("Fuel", "Starts"), ])

state_name_dict = {
    'Battery': ['good', 'bad'],
    'Fuel': ['not empty', 'empty'],
    'Gauge': ['not empty', 'empty'],
    'Turns Over': ['yes', 'no'],
    'Starts': ['yes', 'no'],
}


cpd_b = TabularCPD(variable='Battery', variable_card=2, values=[[0.98], [0.02]], state_names={'Battery': state_name_dict['Battery']})

# p(f = no e), p(f = e)
cpd_f = TabularCPD(variable='Fuel', variable_card = 2, values=[[0.95], [0.05]], state_names={'Fuel': state_name_dict['Fuel']})

# p(g | b, f)
cpd_g = TabularCPD(variable='Gauge', variable_card=2,
                    values=[[0.96, 0.03, 0.9, 0.01], [0.04, 0.97, 0.1, 0.99]],
                    evidence=['Battery', 'Fuel'],
                    evidence_card=[2, 2],
                    state_names={'Gauge': state_name_dict['Gauge'],
                                'Battery': state_name_dict['Battery'],
                                'Fuel': state_name_dict['Fuel']}
                                )
# p(t | b)
cpd_t = TabularCPD(variable='Turns Over', variable_card=2,
                    values=[[0.97, 0.02], [0.03, 0.98]],
                    evidence=['Battery'],
                    evidence_card=[2],
                    state_names={'Turns Over': state_name_dict['Turns Over'],
                                'Battery': state_name_dict['Battery']}
                                )

# p(s)
cpd_s = TabularCPD(variable='Starts', variable_card=2,
                    values=[[0.99, 0.08, 0.0, 0.01], [0.01, 0.92, 1.0, 0.99]],
                    evidence=['Turns Over', 'Fuel'],
                    evidence_card=[2, 2],
                    state_names={'Starts': state_name_dict['Starts'],
                               'Turns Over': state_name_dict['Turns Over'],
                               'Fuel': state_name_dict['Fuel']}
                               )

model.add_cpds(cpd_b, cpd_f, cpd_g, cpd_t, cpd_s)

m = VariableElimination(model)

# Q5.2.
# q5_2 = m.query(['Fuel'], evidence={'Starts': 1})
# print(q5_2)

# Q5.3 Scenario 1
# q5_3_1_b_t = m.query(['Battery'], evidence={'Starts': 1, 'Turns Over'
# : 1})
# q5_3_1_f_t = m.query(['Fuel'], evidence={'Starts': 1, 'Turns Over'
# : 1})
# print(q5_3_1_b_t)
# print(q5_3_1_f_t)

# q5_3_1_b_tg = m.query(['Battery'], evidence={'Starts': 1, 'Turns Over'
# : 1, 'Gauge': 0})
# q5_3_1_f_tg = m.query(['Fuel'], evidence={'Starts': 1, 'Turns Over'
# : 1, 'Gauge': 0})
# print(q5_3_1_b_tg)
# print(q5_3_1_f_tg)

# Q5.3 Scenario 2

q5_3_2_b_g = m.query(['Battery'], evidence={'Starts': 1, 'Gauge'
: 1})
q5_3_2_f_g = m.query(['Fuel'], evidence={'Starts': 1, 'Gauge'
: 1})
print(q5_3_2_b_g)
print(q5_3_2_f_g)

q5_3_2_b_tg = m.query(['Battery'], evidence={'Starts': 1, 'Turns Over'
: 1, 'Gauge': 1})
q5_3_2_f_tg = m.query(['Fuel'], evidence={'Starts': 1, 'Turns Over'
: 1, 'Gauge': 1})
print(q5_3_2_b_tg)
print(q5_3_2_f_tg)