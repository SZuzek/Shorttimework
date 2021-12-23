def solve_process(cc):
    cc.stw_.display = False
    cc.stw_.nt = 201 # need to increase precision since few are affected by stw   
    cc.stw_.init_gekko()
    cc.stw_.solve_gekko(rtol = 1e-07, otol = 1e-07, max_iter = 1000)
    
    return cc
