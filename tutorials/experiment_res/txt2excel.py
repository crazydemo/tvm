import openpyxl
import codecs
from openpyxl.utils import get_column_letter
 
def txt_to_xlsx(filename, sheetname, outfile, first):
 
    fr = codecs.open(filename,'r')
    if first:
        wb = openpyxl.Workbook()
    else:
        wb=openpyxl.load_workbook(outfile)
    ws = wb.active
    ws = wb.create_sheet()
    ws.title = sheetname
    row = 0
    cnt = 0
    for line in fr:
        line = line.strip()
        line = line.split(',')
        row +=1
        col = 0
        for j in range(len(line)):
            col +=1
            print (line[j])
            ws.cell(column = col,row = row,value = line[j].format(get_column_letter(col)))
        
    wb.save(outfile)


def read_xlsx(filename):
    #载入文件
    wb = openpyxl.load_workbook(filename)
    #获取Sheet1工作表
    ws = wb.get_sheet_by_name('Sheet1')
    #按行读取
    for row in ws.rows:
        for cell in row:
            print (cell.value)
    #按列读
    for col in ws.columns:
        for cell in col:
            print (cell.value)
 
if __name__=='__main__':
    first = False
    # inputfileTxt = 'experiment_res/byoc_mobilenet_bs1_raw_pattern_table.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/resnet_bs1.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/resnet_bs64.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/resnet_bs1_byoc.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/resnet_bs64_byoc.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/mobilenet_bs1.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/mobilenet_bs64.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/mobilenet_bs1_byoc.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/mobilenet_bs64_byoc.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs1.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs64.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs1_byoc.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs64_byoc.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs1_byoc_with_gnu.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs64_byoc_with_gnu.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs1_gnu.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/bert_bs64_gnu.txt'
    # inputfileTxt = 'experiment_res/mxnet_onednn/resnet_bs64_byoc_official_table.txt'
    inputfileTxt = 'experiment_res/byoc_test_verbose_time_wo_integration_2.txt'
    outfileExcel = 'experiment_res/byoc_test_verbose_time.xlsx'
    sheetname = inputfileTxt.split('/')[1].split('.')[0]
    txt_to_xlsx(inputfileTxt, sheetname, outfileExcel, first)
    # read_xlsx(outfileExcel)