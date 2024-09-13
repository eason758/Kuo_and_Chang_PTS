import pptx
from colorama import init, Style, Fore
import sys
import numpy as np

init()


def colored(val):
    out = Fore.RED + str(val) + Style.RESET_ALL
    return out 


def print_in_color(arr):
    ans = list()
    for p in arr:
        if p == max(arr):
            print(f"{colored(p):^39}", end='\t')
            ans.append(1)
        else:
            print(f"{p:^30}", end='\t')
            ans.append(0)
    return np.array(ans)


def print_in_double_color(arr1, arr2):
    ans = list()
    for a, b in zip(arr1, arr2):
        text = ''
        offset = 0
        if a == max(arr1):
            text += colored(a)
            offset += 1
        else:
            text += str(a)
        text += '/'
        if b == max(arr2):
            offset += 1
            text += colored(b)
        else:
            text += str(b)
        if offset == 2:
            print(f'{text:^48}', end='\t')
            ans.append(2)
        elif offset == 1:
            print(f'{text:^39}', end='\t')
            ans.append(1)
        else:
            print(f'{text:^30}', end='\t')
            ans.append(0)
    return np.array(ans)


def print_in_triple_color(arr1, arr2, arr3):
    ans = list()
    for a, b, c in zip(arr1, arr2, arr3):
        text = ''
        offset = 0
        if a == max(arr1):
            text += colored(a)
            offset += 1
        else:
            text += str(a)
        text += '/'
        if b == min(arr2):
            offset += 1
            text += colored(b)
        else:
            text += str(b)
        text += '/'
        if c == max(arr3):
            offset += 1
            text += colored(c)
        else:
            text += str(c)
        if offset == 3:
            print(f'{text:^57}', end='\t')
            ans.append(3)
        elif offset == 2:
            print(f'{text:^48}', end='\t')
            ans.append(2)
        elif offset == 1:
            print(f'{text:^39}', end='\t')
            ans.append(1)
        else:
            print(f'{text:^30}', end='\t')
            ans.append(0)
    return np.array(ans)


if __name__ == '__main__':
    ignore_cell = ['Type', 'Top', 'Training', 'Testing', 'Validation', 'Testing', 
                   'Transfer', 'Local', 'Based', 'All', 'Johansen', 'x', 'GMM', 'K-means']
    prs = pptx.Presentation(sys.argv[1])
    # row-based or col-based
    row_col_based = int(sys.argv[2])
    # page number
    slide = prs.slides[int(sys.argv[3])]
    
    for s in slide.shapes:
        if type(s) == pptx.shapes.graphfrm.GraphicFrame:
            all_text_row_based = []
            all_text_col_based = []
            table = s.table
            for r in table.rows:
                row_text = []
                i = 0
                for c in r.cells:
                    tmp = c.text_frame.text
                    count = sum([tmp.startswith(x) for x in ignore_cell])
                    if count == 0 and tmp != '':
                        row_text.append(tmp.split('\n'))
                        if len(all_text_col_based) < i + 1:
                            all_text_col_based.append([tmp.split('\n')])
                        else:
                            all_text_col_based[i].append(tmp.split('\n'))
                        i = i + 1
                if len(row_text) > 0:
                    all_text_row_based.append(row_text)
            print()
            if row_col_based == 0:
                iter_text = all_text_row_based
            elif row_col_based == 1:
                iter_text = all_text_col_based
            else:
                raise ValueError('0 for row-based and 1 for col-based!')
            total_count = None
            for r in iter_text:
                profit = [float(c[0]) for c in r]
                tmp1 = print_in_color(profit)
                print()
                win_rate = [float(c[1].split('%')[0]) for c in r]
                normal_close_rate = [float(c[1].split('%')[1].replace('/', '')) for c in r]
                tmp2 = print_in_double_color(win_rate, normal_close_rate)
                print()
                sharpe_ratio = [float(c[3]) for c in r]
                tmp3 = print_in_color(sharpe_ratio)
                print()
                sharpe_ratio = [float(c[4]) for c in r]
                tmp4 = print_in_color(sharpe_ratio)
                print()
                profit_per_pair = [float(c[5].split('/')[0]) for c in r]
                max_capital = [float(c[5].split('/')[1]) for c in r]
                mdd = [float(c[5].split('/')[2]) for c in r]
                tmp5 = print_in_triple_color(profit_per_pair, max_capital, mdd)
                # print(total_count, tmp1, tmp2, tmp3, tmp4, tmp5)
                if total_count is None:
                    total_count = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
                else:
                    total_count += tmp1 + tmp2 + tmp3 + tmp4 + tmp5
                print('\n')
            print_in_color(total_count)
            print('\n\n\n')
