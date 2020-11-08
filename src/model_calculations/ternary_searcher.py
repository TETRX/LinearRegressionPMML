import math

class TernarySearcher:
    def __init__(self, to_min,error):
        self.to_min=to_min
        self.error=error

    def minimalize(self):
        l,r= self.find_interval(-2*self.error,2*self.error)
        ret=self.ternary_search(l,r)
        return ret

    def find_interval(self,left_border,right_border):
        l=self.to_min(left_border)
        r=self.to_min(right_border)
        sample=(left_border+right_border)/2
        a=self.to_min(sample)
        if not a==min(a,l,r):
            return self.find_interval(2*left_border,2*right_border)
        return (left_border,right_border)

    def ternary_search(self,left_border,right_border):
        if(abs(right_border-left_border)<=self.error):
            return (right_border+left_border)/2

        sample_a=2/3*left_border+1/3*right_border #l-a-b-r#
        sample_b=1/3*left_border+2/3*right_border
        l=self.to_min(left_border)
        r=self.to_min(right_border)
        a=self.to_min(sample_a)
        b=self.to_min(sample_b)
        if l==a and a==b and b==r:
            return (right_border+left_border)/2
        if l<=a and a<=b and b<=r:
            return self.ternary_search(left_border,sample_a)
        if l>=a and a>=b and b>=r: 
            return self.ternary_search(sample_b,right_border)
        if a<=l and a<=b:
            return self.ternary_search(left_border,sample_b)
        if b<=r and b<=l:
            return self.ternary_search(sample_a,right_border)