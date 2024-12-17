def fix_bac_res(res):
    try:
        res = int(res)
        return '0'*(5-len(str(res)))+str(res)
    except:
        return res