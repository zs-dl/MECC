import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mecc import decode, privatize, simplex_projection
from mecc.data import movielens_first, kosarak_clicks
from mecc.full_mle import estimate
from mecc.reports import column_counts


class CoreChecks(unittest.TestCase):
    def test_oue_privacy(self):
        for epsilon in (.5,3.,5.):
            p,q=.5,1/(np.exp(epsilon)+1)
            for y in itertools.product((0,1),repeat=3):
                probabilities=[]
                for x in range(3):
                    probs=np.full(3,q);probs[x]=p
                    probabilities.append(np.prod(np.where(y,probs,1-probs)))
                self.assertLessEqual(max(probabilities)/min(probabilities),np.exp(epsilon)*(1+1e-14))

    def test_reports_and_counts(self):
        values=np.arange(31,dtype=np.int64)%10
        packed=privatize(values,10,3.,seed=2)
        self.assertEqual(packed.shape,(31,2))
        self.assertTrue(np.all(packed[:,-1]>>2==0))
        np.testing.assert_array_equal(packed,privatize(values,10,3.,seed=2))
        np.testing.assert_array_equal(column_counts(packed,10),np.unpackbits(packed,axis=1,count=10,bitorder="little").sum(0))
        with self.assertRaises(ValueError):privatize(np.array([-1]),10,3.)

    def test_control_statistics_and_training_separation(self):
        packed=privatize(np.arange(40)%6,6,3.,seed=5)
        estimates,_,state=decode(packed,40,6,3.)
        bits=np.unpackbits(packed,axis=1,count=6,bitorder="little")
        q=1/(np.exp(3.)+1);z=(bits[state["validation_indices"]]-q)/(.5-q)
        expected=z.T@z[:,state["H"]]/len(z)
        np.testing.assert_allclose(state["T"],expected,rtol=1e-13,atol=1e-13)
        modified=packed.copy();modified[state["validation_indices"]]^=np.uint8(63)
        _,_,state2=decode(modified,40,6,3.)
        np.testing.assert_array_equal(state["beta"],state2["beta"])
        self.assertAlmostEqual(estimates["mecc"].sum(),1.)

    def test_zero_controls_and_empty_control_split(self):
        packed=privatize(np.arange(20)%4,4,1.,seed=4)
        for kwargs in ({"max_controls":0},{"train_fraction":1.0}):
            output,_,_=decode(packed,20,4,1.,**kwargs)
            np.testing.assert_array_equal(output["mecc"],output["calibrate_public"])

    def test_mle_certificate_and_grid(self):
        packed=privatize(np.arange(24)%3,3,2.,seed=6)
        theta,diag=estimate(packed,24,3,2.)
        self.assertTrue(diag["success"])
        self.assertLessEqual(diag["duality_gap"],1e-7)
        bits=np.unpackbits(packed,axis=1,count=3,bitorder="little")
        objective=lambda x:np.log1p(np.expm1(2.)*(bits@x)).mean()
        for i in range(31):
            for j in range(31-i):
                point=np.array([i,j,30-i-j])/30
                self.assertGreaterEqual(objective(theta)+1e-10,objective(point))

    def test_public_data_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"ratings.dat").write_text("2::8::4::9\n1::7::5::10\n1::5::3::10\n2::9::2::8\n")
            np.testing.assert_array_equal(movielens_first(root/"ratings.dat"),[4,8])
            (root/"kosarak.dat").write_text("1 3\n4 2 1\n")
            got=kosarak_clicks(root/"kosarak.dat",3,5)
            np.testing.assert_array_equal(got,np.array([0,2,3,1,0])[np.sort(np.random.Generator(np.random.PCG64(5)).choice(5,3,replace=False))])


if __name__=="__main__":unittest.main()
