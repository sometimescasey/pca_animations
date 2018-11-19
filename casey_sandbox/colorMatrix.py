from big_ol_pile_of_manim_imports import *

def colorMatrix(mtx, width=1, text_scale=1.0):
	
	m = mtx.shape[0]
	n = mtx.shape[1]

	# check if largest value is over 1: if so, normalize grey background
	denom = 1.0
	max_val = np.amax(np.amax(mtx))
	if max_val > 1.0:
		denom = max_val


	s = width # width of each square

	sq_list = []
	sq_num = []

	count=0
	for r in range(m):
		for c in range(n):
			shade = (mtx[r][c]) / denom
			sq = Square(side_length=s, fill_opacity=1, stroke_width=0, fill_color=Color(rgb=(shade, shade, shade))).move_to(r*DOWN*s+c*RIGHT*s)
			sq_list.append(sq)
			
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

	outline = Polygon(top_l, top_r, bottom_r, bottom_l, color=WHITE) 
	# * operator expands lists before the function call
	group = VGroup(outline, *sq_list, *sq_num)
	# Move back to center
	group.move_to(ORIGIN)

	return group