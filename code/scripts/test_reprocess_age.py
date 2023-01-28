

def reprocess_before_plot(x_pos, age_diff_avg, labels):
    len_vec = len(x_pos)
    new_x_pos = x_pos
    new_age_diff_avg = age_diff_avg
    new_labels = labels

    while new_age_diff_avg[0] == 0:
        new_x_pos.pop(0)
        new_age_diff_avg.pop(0)
        new_labels.pop(0)
    while new_age_diff_avg[-1] == 0:
        new_x_pos.pop(-1)
        new_age_diff_avg.pop(-1)
        new_labels.pop(-1)

    return new_x_pos, new_age_diff_avg, new_labels

if __name__ == '__main__':
    x_pos = [0, 1, 2, 3, 4,5, 6, 7, 8, 9]
    age_diff_avg = [0, 0, 5, 16, 12, 34, 67, 0, 0, 0]
    labels =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    new_x_pos, new_age_diff_avg, new_labels = reprocess_before_plot(x_pos, age_diff_avg, labels)
    print(new_x_pos)
    print(new_age_diff_avg)
    print(new_labels)
