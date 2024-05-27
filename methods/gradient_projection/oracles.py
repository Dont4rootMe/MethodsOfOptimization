import numpy as np
import scipy.integrate

class ProjectionOracle:
    def check_inside(self, x: np.array):
        """
        :param x: np.array vectorized function
        :return bool
        """
        return NotImplementedError
    
    def get_projection(self, x: np.array):
        """
        :param x: np.array vectorized function
        :return vectorized projection function
        """
        return NotImplementedError


class SphereProjectionOracle(ProjectionOracle):
    def __init__(self, a: np.array, R: float):
        """
        param: a: np.array vectorised function - center of hypersphere
        param: R: radius of hypersphere in L_2 space. R > 0
        """
        assert scipy.integrate.simpson(a ** 2, dx=1/len(a)) != 0, 'center of hypersphere cant be zero-vector'

        self.a = a
        self.R = R

    def check_inside(self, x: np.array):
        """
        :param x: np.array vectorized function
        :return bool
        """
        assert len(self.a) == len(x), 'wrong space of x or different discriditation'
        return scipy.integrate.simpson((x - self.a) ** 2, dx=1/len(x)) <= self.R ** 2 + 1e-3

    def get_projection(self, x: np.array):
        """
        :param x: np.array vectorized function
        :return vectorized projection function
        """
        assert len(self.a) == len(x), 'wrong space of x or different discriditation'
        return self.a + self.R * (x - self.a) / (scipy.integrate.simpson((x - self.a) ** 2, dx=1/len(x)) ** 0.5)


class SemispaceProjectionOracle(ProjectionOracle):
    def __init__(self, b: np.array, gamma: float):
        """
        param: b: np.array vectorised function - center of hypersphere
        param: gamma: parameter in hypersphere scalar product
        """
        assert scipy.integrate.simpson(b ** 2, dx=1/len(b)) != 0, 'vector direction of semispacecant be zero-vector'

        self.b = b
        self.gamma = gamma

    def check_inside(self, x: np.array):
        """
        :param x: np.array vectorized function
        :return bool
        """
        assert len(self.b) == len(x), 'wrong space of x or different discriditation'
        return scipy.integrate.simpson(x * self.b, dx=1/len(x)) <= self.gamma + 1e-3
    
    def get_projection(self, x: np.array):
        """
        :param x: np.array vectorized function
        :return vectorized projection function
        """
        assert len(self.b) == len(x), 'wrong space of x or different discriditation'
        return x + self.b * (self.gamma - scipy.integrate.simpson(self.b * x, dx=1/len(x))) / scipy.integrate.simpson(self.b ** 2, dx=1/len(x))
    

class ProjectionOracle(ProjectionOracle):
    
    def _check_semispace(self, x):
        assert len(self.b) != len(x), 'wrong space of x or different discriditation'
        return scipy.integrate.simpson(x * self.b, dx=1/len(x)) <= self.gamma
    
    def __init__(self, a_sphere, R, b_hyperplane, gamma):
        assert scipy.integrate.simpson(a_sphere ** 2, dx=1/len(a_sphere)) != 0,         'a vector cant be zero'
        assert scipy.integrate.simpson(b_hyperplane ** 2, dx=1/len(b_hyperplane)) != 0, 'b vector cant be zero'
        assert len(a_sphere) == len(b_hyperplane), 'functions a and b came from different spaces or have different discriditation'
        assert np.abs(gamma) / scipy.integrate.simpson(b_hyperplane ** 2, dx=1/len(b_hyperplane)) ** 0.5 < R, 'hyperplane b does not intersect with hypersphere'

        self.sphere_oracle = SphereProjectionOracle(a_sphere, R)
        self.semispace_oracle = SemispaceProjectionOracle(b_hyperplane, gamma)

        self.a_sphere = a_sphere
        self.center_projection = self.semispace_oracle.get_projection(np.zeros_like(a_sphere)) + a_sphere

    def check_inside(self, x):
        return self.sphere_oracle.check_inside(x) and self.semispace_oracle.check_inside(x - self.a_sphere)
    
    def get_projection(self, x):
        # check if x is already inside
        if self.check_inside(x):
            return x

        sphere_prj = self.sphere_oracle.get_projection(x)
        semspc_prj = self.semispace_oracle.get_projection(x - self.a_sphere) + self.a_sphere

        # if projection of x is on the hypersphere
        if self.semispace_oracle.check_inside(sphere_prj - self.a_sphere):
            return sphere_prj
        
        # if projection of x is on the hyperplane
        if self.sphere_oracle.check_inside(semspc_prj):
            return semspc_prj
        
        # if projection of x is on the intersection of hyperplane and hypersphere
        incoming_vector = self.center_projection - semspc_prj
        norm_inc_vec = scipy.integrate.simpson(incoming_vector ** 2, dx=1/len(x)) ** 0.5
        residual_norm = np.sqrt(self.sphere_oracle.R ** 2 - scipy.integrate.simpson((self.center_projection - self.a_sphere) ** 2, dx=1/len(x)))
        part = 1 - residual_norm / norm_inc_vec
        return semspc_prj + incoming_vector * part
        

class GradientProjectionMethod:
    def __init__(self, oracle: ProjectionOracle, f, grad_f, x0, epsilon, max_iter):
        """
        param: oracle: ProjectionOracle
        param: f: function
        param: grad_f: gradient of function
        param: x0: np.array start point
        param: epsilon: float
        param: max_iter: int
        """
        self.oracle = oracle
        self.f = f
        self.grad_f = grad_f
        self.x = self.oracle.get_projection(x0)
        self.epsilon = epsilon
        self.max_iter = max_iter

    def run(self):
        for i in range(self.max_iter):
            if self.epsilon is not None and scipy.integrate.simpson(self.grad_f(self.x) ** 2, dx=1/len(self.x)) < self.epsilon:
                break
            self.x = self.oracle.get_projection(self.x - self.grad_f(self.x))
        return self.x

    def get_value(self):
        return self.f(self.x)
