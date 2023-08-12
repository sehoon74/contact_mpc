from robot import DynSys

"""
 Single point contact stiffness model with parameters for
 contact location in TCP, rest position, and stiffness.

 This class is _stateless_, just building expressions
   __pars: dict w/ numerical and symbolic parameters for building the expressions
   __vars: DecisionVars for symbolic parameters and associated attributes
"""
class Contact(DynSys):
    def __init__(self, pars, sym_var_attr = {}):
        self.__pars = pars        # Dictionary of the values for the parameters
        self.build_vars(sym_var_attr)

    def build_vars(self, sym_var_attr):
        init = {k: self__pars[k] for k in sym_var_attr.keys()}
        
        self.__vars = DecisionVars(init, **sym_var_attr)
        self.__pars.update(self.__vars)
            
    def get_symdict(self):
        return self.__vars
    
    """
    Return the statedict variables evaluated at the numerical values 
    """
    def get_statedict(self, num_dict):
        fn_input = {k:num_dict[k] for k in self.__vars.keys()}
        return self.statedict_fn(fn_input)

    """
    Build the contact forces and torques
    """
    def build_contact(self, p, R):
        x = p + R@self.__pars['pos']
        disp = x - self.__pars['pos']
        n = self.__pars['stiff']/ca.norm_2(self.__pars['stiff'])
        F = -self.__pars['stiff'].T@(x-self.__pars['rest'])

        fn_dict = dict(p=P, R=R, x=x, disp=disp, n=n, F=F)
        fn_dict.update(self.__vars)
        self.statedict_fn(fn_dict, ['p', 'R', *self.__vars.keys()]
                                   ['x', 'disp', 'n', 'F'])
        
