from big_ol_pile_of_manim_imports import *

# Curly braces, because I don't know what's up with latex
class CurlyBrace:
	def __init__(self, stroke_width=5):
		self.stroke_width = stroke_width
		self.__curly_brace = self.make_curly_brace()

	def brace(self):
		# Return brace object
		return self.__curly_brace

	def make_curly_brace(self):
		b1 = ArcBetweenPoints(UP, DOWN, angle = TAU/3.8, stroke_width=self.stroke_width)
		b1.scale(0.7)
		b1.shift(UP*0.3+RIGHT*0.07)
		b2 = ArcBetweenPoints(UP, UP*2, angle = TAU/4, stroke_width=self.stroke_width)

		bracket = VGroup(b1, b2)
		bracket.rotate(PI/4)
		bracket.rescale_to_fit(3,1,stretch=True)
		bracket.shift(DOWN*3)
		bracket.scale(0.6)
		
		bracket2 = bracket.deepcopy()
		bracket2.flip()
		bracket2.rotate(TAU/2)
		bracket2.shift(UP*1.8)
		
		return VGroup(bracket, bracket2)