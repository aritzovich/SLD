import numpy as np
import pandas as pd



def plot_experiments_logisticRegression(lr= 0.01):

        dataNames= ['haberman', 'mammographic', 'indian_liver', 'heart', 'sonar', 'svmguide3',
                    'liver_disorder', 'german_numer']#'adult', 'iris', 'optdigits', 'satellite', 'vehicle', 'segment', 'redwine', 'letterrecog', 'forestcov', 'ecoli',
        for dataName in dataNames:
            try:
                print(dataName)
                df = pd.read_csv('./Results/results_'+dataName+'_exper_logreg_lr'+str(lr))
                #['seed', 'dataset', 'm', 'n', 'algorithm', 'iter.', 'score', 'type', 'error']

                # Plot the results
                color = 'algorithm'
                style = 'type'
                num_colors = len(df[color].drop_duplicates().values)
                palette = sbn.color_palette("husl", num_colors)

                for score in ["0-1 loss","log. loss"]:  # , "0-1 loss", "rand 0-1 loss"]:
                    print("score: "+str(score))
                    aux = df[(df['dataset'] == dataName)]
                    aux = aux[(aux['score'] == score)]

                    # Calculate the average score for each iteration
                    average_score = aux.groupby(['algorithm', 'iter.', 'type'])['error'].mean().reset_index()
                    print(average_score)

                    # Create the plot
                    sbn.set_style("whitegrid")
                    plt.figure(figsize=(10, 6))

                    # Plot the lines
                    sbn.lineplot(data=aux, x='iter.', y='error', hue='algorithm', style='type', palette= palette, dashes=True)
                    plt.xscale('log')
                    # Set plot title and labels
                    plt.title('Error vs. Iteration')
                    plt.xlabel('Number of Iterations')
                    plt.ylabel('Error')

                    plt.ylim(np.min(aux['error']),np.max(aux[(aux['iter.'] == 0)]['error']))

                    # Show legend
                    plt.legend(title=score+' evolution', bbox_to_anchor=(1.05, 1), loc='upper left')

                    #save the plot
                    plt.savefig("./Results/LogReg_" + dataName + "_" + score + "_" + str(lr) +"_"+ '.pdf', bbox_inches='tight')

                    # Show the plot
                    # plt.show()
            except Exception as e:
                # Handling the exception by printing its description
                print("An exception occurred:", e)

            '''
            fig, ax = plt.subplots(1)
            print("comienzo")
            sbn.lineplot(data=aux, x='iter.', y='error', style=style, hue=color, palette= palette).set_title(dataName + " " + score)
                         #hue_order=["GD", "GDC", "RD"]).set_title(dataName + " " + score)
            ax.set_xscale('log')
            print("medio")
            plt.savefig("./Results/LogReg_" + dataName + "_" + score + "_" + '.pdf', bbox_inches='tight')
            print("fin")
            plt.show()
            '''


def createTable_datasets(name= "./Results/results_exper_QDA_lr0.1.csv"):
    list_to_print= list()
    df= pd.read_csv(name)
    df= df[['data', 'm', 'n']].copy()
    df= df.sort_values(by=["data"])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.insert(0, 'index', df.index)

    print("\\begin{table}[h]\n\centering")
    print(df.to_latex(float_format=lambda x: '{:.3f}'.format(x), index = False))
    print("\caption{Data sets}\n\end{table}")


def createTable_NB(name= "./Results/results_exper_QNB_lr0.1.csv", score_to_print="0-1", threshold= 0):


    ref_to_stop_loss = "0-1"

    df= pd.read_csv(name)
    df= df.sort_values(by=["data","iter"])

    for type in ["ML","MAP"]:
        list_to_print= list()

        df_RD= df[(df["alg"]=="RD") & (df["type"]== type)]
        df_GD= df[(df["alg"]=="GD") & (df["type"]== type)]

        ind= 0
        for _, group in df_RD.groupby(["data"]):
            ind+= 1
            try:
                row=list()
                data= group[group["iter"] == 1]['data'].values[0]
                group_GD= df_GD[df_GD["data"]== data]
                row.append(ind)
                row.append(group[(group["iter"] == 1) & (group["loss"] == score_to_print)]['val'].values[0])
                for iter in range(2,np.max(group["iter"])+1):
                    if (group[(group["iter"] == iter-1) & (group["loss"] == ref_to_stop_loss)]["val"].values[0]
                            - group[(group["iter"] == iter) & (group["loss"] == ref_to_stop_loss)]["val"].values[0]< threshold):
                        row.append(group[(group["iter"] == iter-1) & (group["loss"] == score_to_print)]['val'].values[0])
                        row.append(iter-1)
                        score_RD= group[(group["iter"] == iter-1) & (group["loss"] == score_to_print)]['val'].values[0]
                        iter_RD= iter-1
                        break
                    elif iter == np.max(group["iter"]):
                        row.append(group[(group["iter"] == iter) & (group["loss"] == score_to_print)]['val'].values[0])
                        row.append(iter)
                        score_RD= group[(group["iter"] == iter) & (group["loss"] == score_to_print)]['val'].values[0]
                        iter_RD= iter

                row.append(group_GD[(group_GD["iter"]==iter_RD) & (group_GD["loss"] == score_to_print)]['val'].values[0])
                for iter in range(1,np.max(group_GD["iter"])+1):
                    if group_GD[(group_GD["iter"] == iter) & (group_GD["loss"] == score_to_print)]["val"].values[0]<= score_RD:
                        row.append(iter)
                        break
                    if iter== np.max(group_GD["iter"]):
                        row.append("-")

                list_to_print.append(row)
            except Exception as e:
                # Handling the exception by printing its description
                print(f"Exception {e} in data {data}")


        columns= ["Index", type, "RD", "Iter", "GD", "Reach"]
        df_to_print= pd.DataFrame(list_to_print, columns=columns)
        print("\\begin{table}[h]\n\centering")
        print(df_to_print.to_latex(float_format=lambda x: '{:.3f}'.format(x), index = False))
        print("\caption{Error of discrete NB " + f"with {type} under {score_to_print}" + "}\n\end{table}")



def createTable_QDA(name= "./Results/results_exper_QNB_lr0.1.csv", score_to_print="0-1", threshold= 0):


    ref_to_stop_loss="0-1"
    list_to_print= list()

    df= pd.read_csv(name)
    df['iter'] = df['iter'].astype(int)

    df= df.sort_values(by=["data","iter"])



    ind= 0
    for _, group in df.groupby(["data"]):
        ind+= 1
        try:
            row=list()
            data= group[group["iter"] == 1]['data'].values[0]
            row.append(ind)
            row.append(group[(group["iter"] == 1) & (group["loss"] == score_to_print)]['val'].values[0])
            for iter in range(2,np.max(group["iter"])+1):
                #print(str(group[(group["iter"] == iter-1) & (group["loss"] == "s0-1")]["val"].values[0]))
                if (group[(group["iter"] == iter-1) & (group["loss"] == ref_to_stop_loss)]["val"].values[0]
                        - group[(group["iter"] == iter) & (group["loss"] == ref_to_stop_loss)]["val"].values[0]< threshold):
                    row.append(group[(group["iter"] == iter-1) & (group["loss"] == score_to_print)]['val'].values[0])
                    row.append(iter-1)
                    break
                elif iter == np.max(group["iter"]):
                    row.append(group[(group["iter"] == iter) & (group["loss"] == score_to_print)]['val'].values[0])
                    row.append(iter)

            list_to_print.append(row)
        except Exception as e:
            # Handling the exception by printing its description
            print(f"Exception {e} in data {data}")


    columns= ["Index", "ML", "RD", "Iter"]
    df_to_print= pd.DataFrame(list_to_print, columns=columns)
    df_to_print["Iter"]= df_to_print["Iter"].astype(int)

    print("\\begin{table}[h]\n\centering")
    print(df_to_print.to_latex(float_format=lambda x: '{:.3f}'.format(x), index=False))
    print("\caption{Error of QDA: " + f"{score_to_print}" + "}\n\end{table}")

def createTable_LogReg(name= "./Results/results_exper_LR_lr0.1.csv", score_to_print="0-1", threshold= 0):

    ref_to_stop_loss = "0-1"

    df= pd.read_csv(name)
    df= df.sort_values(by=["data","iter"])

    for type in ["ML","MS"]:
        list_to_print= list()

        df_RD= df[(df["alg"]=="RD") & (df["type"]== type)]
        df_GD= df[(df["alg"]=="GD") & (df["type"]== type)]

        ind= 0
        for _, group in df_RD.groupby(["data"]):
            ind+= 1
            try:
                row=list()
                data= group[group["iter"] == 1]['data'].values[0]
                group_GD= df_GD[df_GD["data"]== data]
                row.append(ind)
                row.append(group[(group["iter"] == 1) & (group["loss"] == score_to_print)]['val'].values[0])
                for iter in range(2,np.max(group["iter"])+1):
                    if (group[(group["iter"] == iter-1) & (group["loss"] == ref_to_stop_loss)]["val"].values[0]
                            - group[(group["iter"] == iter) & (group["loss"] == ref_to_stop_loss)]["val"].values[0]< threshold):
                        row.append(group[(group["iter"] == iter-1) & (group["loss"] == score_to_print)]['val'].values[0])
                        row.append(iter-1)
                        score_RD= group[(group["iter"] == iter-1) & (group["loss"] == score_to_print)]['val'].values[0]
                        iter_RD= iter-1
                        break
                    elif iter == np.max(group["iter"]):
                        row.append(group[(group["iter"] == iter) & (group["loss"] == score_to_print)]['val'].values[0])
                        row.append(iter)
                        score_RD= group[(group["iter"] == iter) & (group["loss"] == score_to_print)]['val'].values[0]
                        iter_RD= iter

                row.append(group_GD[(group_GD["iter"]==iter_RD) & (group_GD["loss"] == score_to_print)]['val'].values[0])
                for iter in range(1,np.max(group_GD["iter"])+1):
                    if group_GD[(group_GD["iter"] == iter) & (group_GD["loss"] == score_to_print)]["val"].values[0]<= score_RD:
                        row.append(iter)
                        break
                    if iter== np.max(group_GD["iter"]):
                        row.append("-")

                list_to_print.append(row)
            except Exception as e:
                # Handling the exception by printing its description
                print(f"Exception {e} in data {data}")


        columns= ["Index", type, "RD", "Iter", "GD", "Reach"]
        df_to_print= pd.DataFrame(list_to_print, columns=columns)
        print("\\begin{table}[h]\n\centering")
        print(df_to_print.to_latex(float_format=lambda x: '{:.3f}'.format(x), index = False))
        print("\caption{Error of LR " + f"with {type} under {score_to_print}" + "}\n\end{table}")


if __name__ == '__main__':
    createTable_datasets()
#    createTable_QDA(name="./Results/results_exper_QDA_lr0.1.csv", score_to_print="0-1")
    createTable_NB(name="./Results/results_exper_NB_lr0.1.csv", score_to_print="0-1")
    createTable_LogReg(name="./Results/results_exper_LR_lr0.1.csv", score_to_print="0-1")
