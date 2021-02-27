p_BT = 0.6
p_MT = 0.2

p_K__BF_MF = 0.3
p_K__BF_MT = 0.2
p_K__BT_MF = 0.6
p_K__BT_MT = 0.1

def find_p_B__K(p_BT, p_MT, p_K__BF_MF, p_K__BF_MT, p_K__BT_MF, p_K__BT_MT):
    top = p_BT * ((p_K__BT_MT) * (p_MT) + (p_K__BT_MF) * (1-p_MT))
    bottom = top + (1-p_BT) * ((p_K__BF_MT) * (p_MT) + (p_K__BF_MF) * (1-p_MT))

    return top/bottom

print(find_p_B__K(p_BT, p_MT, p_K__BF_MF, p_K__BF_MT, p_K__BT_MF, p_K__BT_MT))

new_p_K__BF_MF = 0.01
new_p_K__BF_MT = 0.05
print(find_p_B__K(p_BT, p_MT, new_p_K__BF_MF, new_p_K__BF_MT, p_K__BT_MF, p_K__BT_MT))


def find_p_M__K(p_BT, p_MT, p_K__BF_MF, p_K__BF_MT, p_K__BT_MF, p_K__BT_MT):
    top = p_MT * ((p_K__BT_MT) * (p_BT) + (p_K__BF_MT) * (1-p_BT))
    bottom = top + (1-p_MT) * ((p_K__BT_MF) * (p_BT) + (p_K__BF_MF) * (1-p_BT))

    return top/bottom

print(find_p_M__K(p_BT, p_MT, p_K__BF_MF, p_K__BF_MT, p_K__BT_MF, p_K__BT_MT))
print(find_p_M__K(p_BT, p_MT, new_p_K__BF_MF, new_p_K__BF_MT, p_K__BT_MF, p_K__BT_MT))

print(find_p_M__K(p_BT, p_MT, new_p_K__BF_MF, p_K__BF_MT, p_K__BT_MF, p_K__BT_MT))
print(find_p_M__K(p_BT, p_MT, p_K__BF_MF, new_p_K__BF_MT, p_K__BT_MF, p_K__BT_MT))
