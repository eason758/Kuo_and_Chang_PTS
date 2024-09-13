import pptx
from colorama import init, Style, Fore
import sys

init()

def colored(val):
    out = f'\\textbf{{{val}}}'
    return out 


def print_in_color(arr, text=None, max_min=True, benchmark=False):
    if max_min:
        max_min_func = max
    else:
        max_min_func = min
    print(f"{text:^25}", end=f'\t&')
    for i, p in enumerate(arr):
        if p == arr[-1] and i == len(arr) - 1:
            end_sign = '\\\\\\\\'
        else:
            end_sign = '&'
        if benchmark:
            # Use first two column as benchmark
            if p > max_min_func(arr[:2]) and max_min:
                print(f"{colored(p):^25}", end=f'\t{end_sign}')
            elif p < max_min_func(arr[:2]) and not max_min:
                print(f"{colored(p):^25}", end=f'\t{end_sign}')
            else:
                print(f"{p:^25}", end=f'\t{end_sign}')
        else:
            if p == max_min_func(arr):
                print(f"{colored(p):^25}", end=f'\t{end_sign}')
            else:
                print(f"{p:^25}", end=f'\t{end_sign}')


if __name__ == '__main__':
    prs = pptx.Presentation(sys.argv[1])
    # row-based or col-based
    row_col_based = int(sys.argv[2])
    # page number
    slide = prs.slides[int(sys.argv[3])]
    ignore_cell = ['Type', 'Top', 'Training', 'Testing', 'Validation', 'Testing', 
                   'Transfer', 'Local', 'Based', 'All', 'Johansen', 'x', 'GMM', 'K-means']    
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
            for r in iter_text:
                profit = [float(c[0]) for c in r]
                print_in_color(profit, 'Profit (thousands)')
                print()
                tmp = [c[2] for c in r]
                open_number = list()
                for o in tmp:
                    if ',' in o:
                        open_number.append(o.split(',')[-1].replace(')', ''))
                    elif '/' in o:
                        open_number.append(o.split('/')[-1].replace(')', ''))
                print_in_color(open_number, 'Total open number')
                print()
                win_rate = [float(c[1].split('%')[0]) for c in r]
                print_in_color(win_rate, 'Win rate (\\%)')
                print()
                normal_close_rate = [float(c[1].split('%')[1].replace('/', '')) for c in r]
                print_in_color(normal_close_rate, 'Normal close rate (\\%)')
                print()
                sharpe_ratio = [float(c[3]) for c in r]
                print_in_color(sharpe_ratio, 'Sharpe ratio (daily-based)')
                print()
                sharpe_ratio = [float(c[4]) for c in r]
                print_in_color(sharpe_ratio, 'Sharpe ratio (pair-based)')
                print()
                profit_per_pair = [float(c[5].split('/')[0]) for c in r]
                print_in_color(profit_per_pair, 'Profit per open (thousands)')
                print()
                max_capital = [float(c[5].split('/')[1]) for c in r]
                print_in_color(max_capital, 'Trade capital (thousands)', False)
                print()
                mdd = [float(c[5].split('/')[2]) for c in r]
                print_in_color(mdd, 'MDD')
                print()
                print('\n')
            print('\n\n\n')