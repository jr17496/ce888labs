import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#print(new_fleet)
#print(current_fleet)
#x_new = np.linspace(1,len(new_fleet),len(new_fleet))
#x_current = np.linspace(1,len(current_fleet),len(current_fleet))

#print(len(x_new))
#print(len(new_fleet))

#plt.hist(current_fleet,x_current)
#plt.show()

#plt.hist(new_fleet,x_new)
#plt.show()



#plt.scatter(current_fleet,x_current)
#plt.show()

#plt.scatter(new_fleet,x_new)
#plt.show()

def boostrap(statistic_func, iterations, data):
    samples = np.random.choice(data, replace=True, size=[iterations, len(data)])
    # print samples.shape
    data_mean = data.mean()
    vals = []
    for sample in samples:
        sta = statistic_func(sample)
        # print sta
        vals.append(sta)
    b = np.array(vals)
    # print b
    lower, upper = np.percentile(b, [2.5, 97.5])
    return data_mean, lower, upper

if __name__ == "__main__":

    df = pd.read_csv('vehicles.csv')

    current_fleet_df = df['Current fleet']
    # print(current_fleet)
    new_fleet_df = df['New Fleet']
    new_fleet_df = new_fleet_df.dropna(axis=0, how='any')

    new_fleet = new_fleet_df.values.T
    current_fleet = current_fleet_df.values.T

    standard_diviation_current = np.std(current_fleet)
    standard_diviation_new = np.std(new_fleet)

    data_current = current_fleet
    boots = []
    for i in range(100, 100000, 1000):
        boot = boostrap(np.mean, i, data_current)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

    df_boot_current = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

    data_new = new_fleet
    boots = []
    for i in range(100, 100000, 1000):
        boot = boostrap(np.mean, i, data_new)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

    df_boot_new = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

    ############################################################

    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)

    sns_plot.axes[0,0].set_ylim(0,)
    sns_plot.axes[0,0].set_xlim(0,)

    sns_plot.savefig("scaterplot.png",bbox_inches='tight')



    plt.clf()
    sns_plot2= sns.distplot(current_fleet, bins=20, kde=False, rug=True).get_figure()

    #axes = plt.gca()
    #axes.set_xlabel('Millons of pounds in sales')
    #axes.set_ylabel('Sales count')

    sns_plot2.savefig("histogram.png", bbox_inches='tight')

    plt.clf()
    sns_plot3 = sns.distplot(new_fleet, bins=20, kde=False, rug=True).get_figure()
    sns_plot3.savefig("histogram.png", bbox_inches='tight')

    #########################################################################

    sns_plot = sns.lmplot(df_boot_current.columns[0], df_boot_current.columns[1], data=df_boot_current, fit_reg=False,
                          hue="Value")

    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig("bootstrap_confidence_current.png", bbox_inches='tight')

    ##########################################################################

    sns_plot = sns.lmplot(df_boot_new.columns[0], df_boot_new.columns[1], data=df_boot_new, fit_reg=False,
                          hue="Value")

    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig("bootstrap_confidence_new.png", bbox_inches='tight')

    lower_values_current = []
    upper_values_current = []
    bootstrap_current = df_boot_current.values.T
    for i in range(len(bootstrap_current[0])):
        if bootstrap_current[2][i] == 'lower':
            lower_values_current.append(bootstrap_current[1][i])
        elif bootstrap_current[2][i] == 'upper':
            upper_values_current.append(bootstrap_current[1][i])

    print('lower bound current : ' + str(min(lower_values_current)))
    print('upper bound current : ' + str(max(upper_values_current)))

    lower_values_new = []
    upper_values_new = []
    bootstrap_new = df_boot_new.values.T
    for i in range(len(bootstrap_new[0])):
        if bootstrap_new[2][i] == 'lower':
            lower_values_new.append(bootstrap_new[1][i])
        elif bootstrap_new[2][i] == 'upper':
            upper_values_new.append(bootstrap_new[1][i])

    print('lower bound new : ' + str(min(lower_values_new)))
    print('upper bound new : ' + str(max(upper_values_new)))
