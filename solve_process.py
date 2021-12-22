def solve_process(cc):
    cc.stw_.display = False
    cc.stw_.nt = 4001 # need to increase precision since few are affected by stw   
    cc.stw_.init_gekko()
    cc.stw_.solve_gekko()
    
    return cc
