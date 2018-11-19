from big_ol_pile_of_manim_imports import *

class Shapes(Scene):
	# Just make some simple shapes

	def construct(self):
		circle = Circle()
		movedCircle = Circle().move_to(UP+LEFT)
		
		self.add(circle)
		self.play(Transform(circle, movedCircle))
