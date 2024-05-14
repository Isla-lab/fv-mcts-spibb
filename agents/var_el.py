import numpy as np

class Function:
    def __init__(self, variables, Q, node, exploration_bonus=True) -> None:
        self._variables = variables
        self._positions = variables
        self._conditioned_variables = variables
        self._Q = Q
        self._node = node
        self._argmax = dict()

    def compute(self, args=None):
        if args is None:
            args = dict()
        assert len(self.conditioned_variables) == 2
        key0 = self.conditioned_variables[0]
        key1 = self.conditioned_variables[1]
        val0 = args[key0]
        val1 = args[key1]
        return self._Q[(key0, key1)][val0][val1]
    
    @property
    def variables(self):
        return self._variables
    
    @property
    def conditioned_variables(self):
        return self._conditioned_variables

    @property
    def argmax(self):
        return self._argmax

class MaxFunction(Function):
    def __init__(self, variables, positions, _conditioned_variables, functions, possible_values, node, exploration_bonus=True) -> None:
        super().__init__(variables, positions, _conditioned_variables, node, exploration_bonus)
        self._functions = functions
        self._possible_values = possible_values

    def __init__(self, conditioned_variables, functions, possible_values, node, exploration_bonus=True) -> None:
        self._variables = []
        self._functions = []
        self._positions = []
        self._possible_values = possible_values
        self._node = node
        self._exploration_bonus = exploration_bonus
        self._argmax = dict()
        self._conditioned_variables = conditioned_variables
        for f in functions:
            var = list(np.array(f.variables).flatten())
            for v in var:
                if v not in self._conditioned_variables and v not in self._positions:
                    self._positions.append(v)
            self._functions.append(f)
            arr = np.array(f.variables)
            if len(arr.shape) > 1:
                for subarr in arr:
                    self._variables.append(list(subarr))
            else:
                self._variables.append(f.variables)

    def compute(self, args=None):
        if args is None:
            args = dict()
        # Maximize, so for each value of the free variable compute
        cv = self._conditioned_variables[0]
        bonuses = [0 for _ in self._possible_values[cv]]
        if self._exploration_bonus:
            node = self._node
            i = self.conditioned_variables[0]
            bonuses = np.array(node.param.C * np.sqrt(
                np.divide(np.log((node.ns + 1)), node.N[i],
                        out=np.full_like(node.N[i], np.inf),
                        where=node.N[i] != 0)
                ))
        # For each possible value of the variable
        max = float("-inf")
        argmax = -1
        for j, value in enumerate(self._possible_values[cv]):
            # Set up the arguments
            args[cv] = value
            # Counter
            res = 0
            # Compute the sum over its functions
            for f in self._functions:
                res += f.compute(args=args)
            
            # Add bonus. If expl was False it is zero
            res += bonuses[j]
            
            if res > max or (res == max and np.random.rand() <= 0.5):
                max = res
                argmax = value
                for f in self._functions:
                    for a in f.argmax:
                        self._argmax[a] = f.argmax[a]
            
        self._argmax[self.conditioned_variables[0]] = argmax
        return max
    
    def compute_argmax(self):
        self.compute()
        return self._argmax
           

class FunctionBuilder:
    def __init__(self, indices, order, Q, possible_values, node, exploration_bonus=True, heuristic=False) -> None:
        self._indices = indices
        # Min_neigh heuristic
        if heuristic:
            self._order = self.compute_best_order(indices, np.array(order).max()+1)
        else:
            self._order = order
        self._possible_values = possible_values
        self._Q = Q
        self._node = node
        self._exploration_bonus = exploration_bonus
        self._functions = []

    # Using Min_neigh heuristic
    def compute_best_order(self, indices, max):
        order = []
        remaining = list(range(max))
        new_indices = [i for i in indices]
        while len(remaining) >= 1:
            min = float("inf")
            argmin = -1
            touched = []
            for i in remaining:
                c = 0
                ti = []
                t = []
                for j, i_ in enumerate(new_indices):
                    if i in i_:
                        c += 1
                        ti.append(j)
                        t.append(i_)
                if c < min:
                    min = c
                    argmin = i
                    touched = t
            # Change data given we have chosen i
            order.append(argmin)
            remaining.remove(argmin)
            # New indices
            new_i = []
            for t in touched:
                for k in t:
                    if k not in new_i:
                        new_i.append(k)
                new_indices.remove(t)
            new_indices.append(new_i)
        return order
    
    def create_functions(self):
        for i in self._indices:
            self._functions.append(Function([i[0], i[1]], self._Q, self._node, self._exploration_bonus))
    
    def create_max_function(self):
        self.create_functions()
        assert(len(self._functions) != 0)
        for var in self._order:
            functions_with_var = []
            for f in self._functions:
                variables = list(np.array(f.variables).flatten())
                if var in variables:
                    functions_with_var.append(f)
            for f in functions_with_var:
                self._functions.remove(f)
            # Now build the function
            self._functions.append(MaxFunction([var], functions_with_var, self._possible_values, self._node, self._exploration_bonus))
        assert(len(self._functions) == 1)
        return self._functions[0]

def test():

    possible_values = {0: [0,1], 1: [0,1], 2: [0,1]}
    # -- Just create some Q -- #
    Q = dict()
    for i in range(3):
        for j in range(3):
            Q[(i,j)] = dict()
            Q[(i,j)][0] = dict()
            Q[(i,j)][1] = dict()
            Q[(i,j)][0][0] = 0
            Q[(i,j)][1][0] = i
            Q[(i,j)][0][1] = j
            Q[(i,j)][1][1] = i+j

    Q[(0,2)][1][0] = 12
    Q[(1,1)][0][0] = -5
    # ------------------------ #

    a0 = Function([0,2], Q, None)
    a1 = Function([0,1], Q, None)
    a2 = Function([1,2], Q, None)

    order = [0, 1, 2]

    f = FunctionBuilder([[0,2], [0,1], [1,2]], order, Q, possible_values,  None, exploration_bonus=False, heuristic=False).create_max_function()

    f.compute_argmax()

    print(f"f: {f.argmax}")

if __name__ == "__main__":
    test()