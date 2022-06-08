from tabulate import tabulate
# tabulate.LATEX_ESCAPE_RULES = {}


def  latex_table(epsilon_list, leng_epsilon_list, latex_code_output_file_name,
                correct_pred_mean_auer, correct_pred_mean_robust,
                total_samp_mean_auer, total_samp_mean_robust,
                ratio_of_opt_pred_to_tot_pred_mean_auer,  ratio_of_opt_pred_to_tot_pred_mean_robust,
                pred_arms_violate_sc2_mean_auer, pred_arms_violate_sc2_mean_robust):


    header_list= [r"", r"Attack Prob.", r"PSR", r"AS", r"RO", r"VSC"]
    flat_list_of_lists= list()
    flat_list_of_lists.append(header_list)
    flat_list_of_lists.append(["","", "", "", "", ""])
    flat_list_of_lists.append(["R-PSI","", "", "", "", ""])
    for i in range(leng_epsilon_list):
        flat_list_robust = [r"", epsilon_list[i] , correct_pred_mean_robust[i], total_samp_mean_robust[i],
                            ratio_of_opt_pred_to_tot_pred_mean_robust[i], pred_arms_violate_sc2_mean_robust[i]]

        flat_list_of_lists.append(flat_list_robust)

    flat_list_of_lists.append(["","", "", "", "", ""])
    flat_list_of_lists.append(["Auer", "", "", "", "", ""])
    for i in range(leng_epsilon_list):
        flat_list_auer = [r"",epsilon_list[i] ,correct_pred_mean_auer[i], total_samp_mean_auer[i],
                          ratio_of_opt_pred_to_tot_pred_mean_auer[i], pred_arms_violate_sc2_mean_auer[i]]
        flat_list_of_lists.append(flat_list_auer)

    print(tabulate(flat_list_of_lists, tablefmt="plain") )

    with open(latex_code_output_file_name, 'w') as f:
        print(tabulate(flat_list_of_lists, headers="firstrow", tablefmt="plain"), file = f)

