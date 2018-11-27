from big_ol_pile_of_manim_imports import *
from PIL import Image
import time

class ColorMatrix:
	def __init__(self, mtx, text_scale=1.0, no_text=False, norm_colors=False, tint='K'):
		self.__mtx = self.make_mtx_group(mtx, text_scale, no_text, norm_colors, tint)

	def mtx(self):
		return self.__mtx

	def make_mtx_group(self, mtx, text_scale, no_text, norm_colors, tint):
		m = mtx.shape[0]
		n = mtx.shape[1]

		# check if largest value is over 1: if so, normalize grey background
		denom = 1.0
		max_val = np.amax(np.amax(mtx))
		min_val = np.amin(np.amin(mtx))
		val_range = max_val - min_val
		if max_val > 1.0:
			denom = max_val


		s = 1 # width of each square

		sq_list = []
		sq_num = []

		count=0
		for r in range(m):
			for c in range(n):
				if norm_colors:
					normalized = float(mtx[r][c] - min_val) / val_range
					shade = normalized
				else:
					shade = (mtx[r][c]) / denom
					shade = max(0, shade) # hack to prevent negative values from messing things up
					
				if tint=='R':
					print("tint is R")
					color_base = [1, 0, 0]
				elif tint=='G': 
					color_base = [0, 1, 0]
				elif tint=='B':
					color_base = [0, 0, 1]
				else:
					color_base = [1, 1, 1] # default black

				color_prod = np.multiply(color_base, shade)
				
				color = Color(rgb=(color_prod[0], color_prod[1], color_prod[2]))
				sq = Square(side_length=s, fill_opacity=1, stroke_width=0, fill_color=color).move_to(r*DOWN*s+c*RIGHT*s)
				sq_list.append(sq)
				
				if (not no_text):
					# automatically flip between white and black text for labels
					text_color = Color(rgb=(0,0,0))
					if (shade < 0.5):
						text_color = Color(rgb=(1,1,1))

					label = TextMobject(str(mtx[r][c]),color=text_color,background_stroke_width=0).move_to(r*DOWN*s+c*RIGHT*s)
					label.scale(text_scale)
					sq_num.append(label)
				# self.add(sq_list[count], sq_num[count])
				count += 1

		# get_corner uses UP LEFT, not TOP LEFT
		top_l = sq_list[0].get_corner(UP+LEFT)
		top_r = top_l + n*s*RIGHT
		bottom_l = top_l + m*s*DOWN
		bottom_r = bottom_l + n*s*RIGHT

		outline = Polygon(top_l, top_r, bottom_r, bottom_l, color=WHITE, stroke_width=2) 
		# * operator expands lists before the function call
		if (no_text):
			group = VGroup(*sq_list, outline)
		else:
			group = VGroup(*sq_list, *sq_num, outline)
		# Move back to center
		group.move_to(ORIGIN)

		return group

class ColorMatrixSkew:
	def __init__(self, mtx, skew=1, shadow=False):
		self.__mtx = self.make_mtx_group(mtx, skew, shadow)

	def mtx(self):
		return self.__mtx

	def make_mtx_group(self, mtx, skew, shadow):
		m = mtx.shape[0]
		n = mtx.shape[1]

		# check if largest value is over 1: if so, normalize grey background
		denom = 1.0
		max_val = np.amax(np.amax(mtx))
		if max_val > 1.0:
			denom = max_val


		s = 1 # width of each square

		sq_list = []
		sq_num = []

		count=0
		for r in range(m):
			for c in range(n):
				shade = (mtx[r][c]) / denom
				sq = Square(side_length=s, fill_opacity=1, stroke_width=0, fill_color=Color(rgb=(shade, shade, shade))).move_to(r*DOWN*s+c*RIGHT*s)
				
				color = Color(rgb=(shade, shade, shade))
				a = sq.get_corner(TOP+LEFT)+(LEFT*(skew))
				b = sq.get_corner(TOP+RIGHT)
				c = sq.get_corner(BOTTOM+RIGHT)+(RIGHT*(skew))
				d = sq.get_corner(BOTTOM+LEFT)
				

				fake_sq = Polygon(a,b,c,d, fill_color=color, fill_opacity=1, stroke_width=0).move_to(r*DOWN*s+c*RIGHT*(1+skew)*s).shift(r*skew*RIGHT)
				
				sq_list.append(fake_sq)
				count += 1

		# get_corner uses UP LEFT, not TOP LEFT
		top_l = sq_list[0].get_corner(UP+LEFT)
		top_r = top_l + n*s*RIGHT*(1+skew)
		bottom_l = top_l + m*s*DOWN + RIGHT*m*skew
		bottom_r = bottom_l + n*s*RIGHT*(1+skew)

		outline = Polygon(top_l, top_r, bottom_r, bottom_l, color=WHITE, stroke_width=1) 
		if shadow:
			# TODO: make shadow distance, alpha a settable param
			shadow_box = Polygon(top_l, top_r, bottom_r, bottom_l, 
				fill_color=BLACK, fill_opacity=0.5, stroke_width=0).shift(DOWN+LEFT*0.3)
			group = VGroup(shadow_box, outline, *sq_list)

		else:
			group = VGroup(*sq_list, outline)
		
		# Move back to center
		group.move_to(ORIGIN)

		return group

def imgToMtx(filename):
	# Helper function to take any image and return it as a np.array 
	# to be used in colorMatrix or colorMatrixSkew
	return np.asarray(Image.open(filename))
	