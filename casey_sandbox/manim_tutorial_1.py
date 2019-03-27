from big_ol_pile_of_manim_imports import *
from once_useful_constructs.matrix_multiplication import *
from casey_sandbox.colorMatrix import *
from casey_sandbox.cov_mtx_6 import cov_mtx_6, eig_mtx_6
from casey_sandbox.miscParts import *
import time

import os

class Shapes(Scene):
	# Just make some simple shapes

	def construct(self):
		circle = Circle()
		movedCircle = Circle().move_to(UP+LEFT)
		
		self.add(circle)
		self.play(Transform(circle, movedCircle))

class AddingText(Scene):
	def construct(self):
		my_first_text = TextMobject("Writing math with manim is fun")
		second_line = TextMobject("and easy to do")
		second_line.next_to(my_first_text, DOWN)
		third_line = TextMobject("for me and you!")
		third_line.next_to(my_first_text, DOWN)

		self.add(my_first_text, second_line)
		self.wait(2)
		self.play(Transform(second_line, third_line))
		self.wait(2)
		self.play(ApplyMethod(second_line.to_corner, UP+LEFT), ApplyMethod(my_first_text.shift, 3*UP))

class RotateAndHighlight(Scene):
	#Rotation of text and highlighting with surrounding geometries
	def construct(self):
		square=Square(side_length=5,fill_color=YELLOW, fill_opacity=1)
		label=TextMobject("Text at an angle")
		label.bg=BackgroundRectangle(label,fill_opacity=1)
		label_group=VGroup(label.bg,label) #Order matters
		label_group.rotate(TAU/8)
		label2=TextMobject("Boxed text",color=BLACK)
		label2.bg=SurroundingRectangle(label2,color=BLUE,fill_color=RED, fill_opacity=.5)
		label2_group=VGroup(label2,label2.bg)
		label2_group.next_to(label_group,DOWN)
		label3=TextMobject("Rainbow")
		label3.scale(2)
		label3.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)
		label3.to_edge(DOWN)
		 
		self.add(square)
		self.play(FadeIn(label_group))
		self.play(FadeIn(label2_group))
		self.play(FadeIn(label3))

class BasicEquations(Scene):
	# A short script showing how to use LaTeX commands
	def construct(self):
		eq1 = TextMobject("$\\vec{X}_0 \\cdot \\vec{Y}_1 = 3$")
		eq1.shift(2*UP)

		eq2 = TextMobject("$\\vec{F}_{net} = \\sum_i \\vec{F}_i$")
		eq2.shift(2*DOWN)

		self.play(Write(eq1))
		self.play(Write(eq2))

class ColoringEquations(Scene):
#Grouping and coloring parts of equations
	def construct(self):
		line1=TexMobject("\\text{The vector }", "\\vec{F}_{net}", "\\text{ is the net }", "\\text{force on object of mass }")
		line1.set_color_by_tex("on object of mass", BLUE)
		line2=TexMobject("m", "\\text{ and acceleration }", "\\vec{a}", ". ")
		line2.set_color_by_tex_to_color_map({
		"m": YELLOW,
		"{a}": RED,
		"acceleration": PINK,
		})
		sentence=VGroup(line1,line2)
		sentence.arrange_submobjects(DOWN, buff=MED_LARGE_BUFF)
		self.play(Write(sentence))

class UsingBraces(Scene):
#Using braces to group text together
	def construct(self):
		eq1A = TextMobject("4x + 3y")
		eq1B = TextMobject("=")
		eq1C = TextMobject("0")
		eq2A = TextMobject("5x -2y")
		eq2B = TextMobject("=")
		eq2C = TextMobject("3")
		eq1B.next_to(eq1A,RIGHT)
		eq1C.next_to(eq1B,RIGHT)
		eq2A.shift(DOWN)
		eq2B.shift(DOWN)
		eq2C.shift(DOWN)
		eq2A.align_to(eq1A,LEFT)
		eq2B.align_to(eq1B,LEFT)
		eq2C.align_to(eq1C,LEFT)

		eq_group=VGroup(eq1A,eq2A)
		braces=Brace(eq_group,LEFT)
		eq_text = braces.get_text("A pair of equations")

		self.add(eq1A, eq1B, eq1C)
		self.add(eq2A, eq2B, eq2C)
		self.play(GrowFromCenter(braces),Write(eq_text))

class UsingBracesConcise(Scene):
#A more concise block of code with all columns aligned
	def construct(self):
		eq1_text=["4","x","+","3","y","=","0"]
		eq2_text=["5","x","-","2","y","=","3"]
		eq1_mob=TexMobject(*eq1_text)
		eq2_mob=TexMobject(*eq2_text)
		eq1_mob.set_color_by_tex_to_color_map({
		"x":RED_B,
		"y":GREEN_C
		})
		eq2_mob.set_color_by_tex_to_color_map({
		"x":RED_B,
		"y":GREEN_C
		})
		for i,item in enumerate(eq2_mob):
			item.align_to(eq1_mob[i],LEFT)
		eq1=VGroup(*eq1_mob)
		eq2=VGroup(*eq2_mob)
		eq2.shift(DOWN)
		eq_group=VGroup(eq1,eq2)
		braces=Brace(eq_group,LEFT)
		eq_text = braces.get_text("A pair of equations")

		self.play(Write(eq1),Write(eq2))
		self.play(GrowFromCenter(braces),Write(eq_text))

class PlotFunctions(GraphScene):
	CONFIG = {
	"x_min" : -10,
	"x_max" : 10,
	"y_min" : -1.5,
	"y_max" : 1.5,
	"graph_origin" : ORIGIN ,
	"function_color" : RED ,
	"axes_color" : GREEN,
	"x_labeled_nums" :range(-10,12,2),

	}
	def construct(self):
		self.setup_axes(animate=True)
		func_graph=self.get_graph(self.func_to_graph,self.function_color)
		func_graph2=self.get_graph(self.func_to_graph2)
		vert_line = self.get_vertical_line_to_graph(TAU,func_graph,color=YELLOW)
		graph_lab = self.get_graph_label(func_graph, label = "\\cos(x)")
		graph_lab2=self.get_graph_label(func_graph2,label = "\\sin(x)", x_val=-10, direction=UP/2)
		two_pi = TexMobject("x = 2 \\pi")
		label_coord = self.input_to_graph_point(TAU,func_graph)
		two_pi.next_to(label_coord,RIGHT+UP)

		self.play(ShowCreation(func_graph),ShowCreation(func_graph2))
		self.play(ShowCreation(vert_line), ShowCreation(graph_lab), ShowCreation(graph_lab2),ShowCreation(two_pi))

	def func_to_graph(self,x):
		return np.cos(x)

	def func_to_graph2(self,x):
		return np.sin(x)

class ExampleApproximation(GraphScene):
	CONFIG = {
	"function" : lambda x : np.cos(x),
	"function_color" : BLUE,
	"taylor" : [lambda x: 1, lambda x: 1-x**2/2, lambda x: 1-x**2/math.factorial(2)+x**4/math.factorial(4), lambda x: 1-x**2/2+x**4/math.factorial(4)-x**6/math.factorial(6),
	lambda x: 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6)+x**8/math.factorial(8), lambda x: 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6)+x**8/math.factorial(8) - x**10/math.factorial(10)],
	"center_point" : 0,
	"approximation_color" : GREEN,
	"x_min" : -10,
	"x_max" : 10,
	"y_min" : -1,
	"y_max" : 1,
	"graph_origin" : ORIGIN ,
	"x_labeled_nums" :range(-10,12,2),

	}
	def construct(self):
		self.setup_axes(animate=True)
		func_graph = self.get_graph(
		self.function,
		self.function_color,
		)
		approx_graphs = [
		self.get_graph(
		f,
		self.approximation_color
		)
		for f in self.taylor
		]

		term_num = [
		TexMobject("n = " + str(n),aligned_edge=TOP)
		for n in range(0,8)]
		[t.to_edge(BOTTOM,buff=SMALL_BUFF) for t in term_num]

		term = TexMobject("")
		term.to_edge(BOTTOM,buff=SMALL_BUFF)

		approx_graph = VectorizedPoint(
		self.input_to_graph_point(self.center_point, func_graph)
		)

		self.play(
		ShowCreation(func_graph),
		)
		for n,graph in enumerate(approx_graphs):
			self.play(
			Transform(approx_graph, graph, run_time = 2),
			Transform(term,term_num[n])
			)
			self.wait()

class ExampleThreeD(ThreeDScene):
	CONFIG = {
	"plane_kwargs" : {
	"color" : RED_B
	},
	}

	def construct(self):
		line1 = Arrow(np.array([0,0,0]), np.array([1,1,1]), color=RED)
		line2 = Arrow(np.array([0,0,0]), np.array([2,2,2]), color=RED)

		line3 = Arrow(np.array([0,0,0]), np.array([-1,-1,-1]), color=RED)
		line4 = Arrow(np.array([0,0,0]), np.array([-2,-2,-2]), color=RED)

		self.set_camera_orientation(0, -np.pi/2)
		plane = NumberPlane(**self.plane_kwargs)
		plane.main_lines.fade(.9)
		plane.add(plane.get_axis_labels())
		self.add(plane)

		self.wait()
		self.move_camera(0.8*np.pi/2, -0.45*np.pi)
		grow_shrink = 10

		self.begin_ambient_camera_rotation()

		# self.add(line1)

		i = 100
		while(i > 0):
			self.play(TransformFromCopy(line1, line2), TransformFromCopy(line3, line4))
			self.wait()
			self.remove(line2, line4)
			self.play(TransformFromCopy(line2, line1), TransformFromCopy(line4, line3))
			self.wait()
			self.remove(line1, line3)
			i -= 1

		self.wait(60)

class TestMatrix(Scene):
	def construct(self):
		basis = RoundedRectangle(height=3, width=0.7, corner_radius=0.2).move_to(DOWN*0.81+LEFT*1.65)

		cov_mtx_0 = [[ 0.231,  0.172,  0.151,  0.135,  0.003,  0.148,  0.122,  0.172,  0.137],
					[ 0.172,  0.284,  0.178,  0.158,  0.002,  0.174,  0.14 ,  0.187,  0.142],
					[ 0.151,  0.178,  0.243,  0.162, -0.005,  0.154,  0.144,  0.172,  0.122],
					[ 0.135,  0.158,  0.162,  0.229, -0.006,  0.136,  0.106,  0.144,  0.125],
					[ 0.003,  0.002, -0.005, -0.006,  0.099, -0.002,  0.002,  0.005, -0.001],
					[ 0.148,  0.174,  0.154,  0.136, -0.002,  0.231,  0.135,  0.179,  0.132],
					[ 0.122,  0.14 ,  0.144,  0.106,  0.002,  0.135,  0.196,  0.138,  0.1  ],
					[ 0.172,  0.187,  0.172,  0.144,  0.005,  0.179,  0.138,  0.26 ,  0.147],
					[ 0.137,  0.142,  0.122,  0.125, -0.001,  0.132,  0.1  ,  0.147,  0.201]]

		# requires list of strings
		matrix = Matrix([["1","2"],["3","4"]]).shift(2*RIGHT)

		self.add(matrix)
		self.wait(1)
		mtx = np.asarray([[1,0,1],[0.5,1,2],[3,4,5]])
		mtx_flat = np.asarray([[1,0,1],[0.5,1,2],[3,4,5]]).reshape(9,1)
		matrix_group = colorMatrix(mtx, text_scale=0.5, no_text=False).mtx().shift(2*LEFT)
		matrix_group2 = colorMatrix(mtx_flat, text_scale=0.5, no_text=False).mtx().shift(4*LEFT)
		matrix_group3 = colorMatrix(mtx, text_scale=0.5, no_text=False).mtx().shift(2*LEFT)

		self.add(matrix_group)
		self.wait(1)
		self.play(ReplacementTransform(matrix_group, matrix_group2))
		self.wait()
		self.play(Transform(matrix_group2, matrix_group3))
		self.wait()

class Black_to_white(Scene):
	def construct(self):
		## Each pixel can range from 0 for white to 1 for black
		# -------------------
		zero = TextMobject("0")
		to = TextMobject("to").next_to(zero)
		one = TextMobject("1", background_stroke_color=WHITE, color=BLACK).next_to(to)
		# to.set_color_by_gradient(WHITE, BLACK)
		rect_1 = Rectangle(width=0.15, height=0.15, fill_color=WHITE, fill_opacity=1).next_to(zero).shift(UP*0.5+LEFT*0.42)
		rect_2 = Rectangle(width=0.2, height=0.2, fill_color=BLACK, 
			fill_opacity=1, background_stroke_color=WHITE, 
			stroke_width=2).next_to(zero).shift(UP*0.5+RIGHT*0.64)
		arrow = Arrow(start=zero, end=one)
		
		zero_to_one = VGroup(zero, arrow, rect_1, one)
		self.add(zero_to_one)
		self.wait(1)
		self.play(Transform(rect_1, rect_2))
		self.wait(2)

class Intro(Scene):
	def construct(self):
		# Title
		# -------------------
		pca = TextMobject("Principal Component Analysis (PCA)").move_to(TOP+DOWN*1)
		pca.scale(1)
		self.add(pca)

		# ## Intro Images
		# -------------------

		cols = 10
		rows = 3
		total = cols*rows

		digits = [None] * cols * rows
		ddd = [None] * cols
		last_row = [None] * cols
 
		img_scale = 0.4

		sixes = [None] * 5

		count = 0
		for i in range(0,rows):
			for j in range(0,cols):
				digits[count] = ImageMobject("casey_sandbox/scenes/1_intro/assets/" + str(j) + ".0_" + str(i) +".png", 
					invert=False, image_mode="RGBA").shift((j-4)*RIGHT*1.2+(i-1)*DOWN*1.2).shift(LEFT*0.5)
				digits[count].scale(0.4)
				if j==6:
					sixes[i] = digits[count]
				count += 1
			
		for i in range(0,cols):
			ddd[i] = TextMobject("...").next_to(digits[total-i-1]).shift(DOWN*2*img_scale+LEFT*2*img_scale)
			if i == 3:
				sixes[rows] = ddd[i]

		for j in range(0,cols):
			last_row[j] = ImageMobject("casey_sandbox/scenes/1_intro/assets/" + str(j) + ".0_" + str(rows) +".png", 
					invert=False, image_mode="RGBA").shift((j-4)*RIGHT*1.2+(rows-1)*DOWN*1.5).shift(LEFT*0.5)
			last_row[j].scale(0.4)
			if j==6:
				sixes[rows+1] = last_row[j]
				
		self.add(*digits, *ddd, *last_row)
		sixes_group = VGroup(*sixes)
		self.wait(2)

		# ## Each is 8x8 = 64
		# -------------------
		a = digits[0].get_corner(UP+LEFT)+UP*0.25
		b = digits[0].get_corner(DOWN+LEFT)+DOWN*0.25
		c = digits[0].get_corner(UP+LEFT)+LEFT*0.25
		d = digits[0].get_corner(UP+RIGHT)+RIGHT*0.25
		eight_v = DoubleArrow(start=a, end=b, tip_length=0.1, tip_width_to_length_ratio=1, color=YELLOW).shift(0.2*LEFT)
		eight_v_l = TextMobject("8px", color=YELLOW).next_to(eight_v).shift(LEFT*1.1)
		eight_v_l.scale(0.7)
		eight_h = DoubleArrow(start=c, end=d, tip_length=0.1, tip_width_to_length_ratio=1, color=YELLOW).shift(0.2*UP)
		eight_h_l = TextMobject("8px", color=YELLOW).next_to(eight_h.get_corner(TOP+LEFT)).shift(UP*0.3+LEFT*0.2)
		eight_h_l.scale(0.7)
		
		eight_by_eight = VGroup(eight_v, eight_v_l, eight_h, eight_h_l)
		self.play(FadeIn(eight_by_eight))

		sixtyfour = TextMobject("= 64 dimensions", color=YELLOW).next_to(digits[1].get_corner(TOP+LEFT)).shift(UP*0.5+LEFT*0.3)
		sixtyfour.scale(0.9)

		# = 64 dimensions
		self.play(FadeIn(sixtyfour))

		self.wait(1)
		self.play(FadeOut(sixtyfour), FadeOut(eight_by_eight), FadeOut(pca))

		
		## Each pixel can range from 0 for white to 1 for black
		# -------------------
		zero = TextMobject("0").next_to(digits[1].get_corner(TOP+LEFT)).shift(UP*0.5+LEFT*0.3)
		to = TextMobject("to").next_to(zero)
		one = TextMobject("1", background_stroke_color=WHITE, color=BLACK).next_to(to)
		# to.set_color_by_gradient(WHITE, BLACK)
		rect_1 = Rectangle(width=0.15, height=0.15, fill_color=WHITE, fill_opacity=1).next_to(zero).shift(UP*0.5+LEFT*0.42)
		rect_2 = Rectangle(width=0.2, height=0.2, fill_color=BLACK, 
			fill_opacity=1, background_stroke_color=WHITE, 
			stroke_width=2).next_to(zero).shift(UP*0.5+RIGHT*0.64)
		arrow = Arrow(start=zero, end=one)
		
		zero_to_one = VGroup(zero, arrow, rect_1, one)
		self.wait(1)
		self.play(FadeInFrom(zero_to_one, direction=LEFT))
		self.wait(1)
		self.play(Transform(rect_1, rect_2))
		self.wait(2)



		## 255 disclaimer
		# ----------
		l1 = TextMobject("* This is reversed from the usual convention where black is 0 and white is 255, but the math doesn't change. We're just flipping the scale backwards and dividing the value by 255.", color=WHITE).next_to(one).shift(LEFT*3.5+UP*0.5) 
		l1.scale(0.5)
		# l2 = TextMobject("", color=WHITE).next_to(l1, DOWN).shift(UP*0.5)
		# l2.scale(0.5)

		disclaimer = VGroup(l1)
		self.add(disclaimer)
		self.wait(1)
		self.remove(disclaimer)
		self.wait(1)

		self.play(FadeOut(zero_to_one), FadeOut(rect_2))
		self.wait()

		# Green box around sixes
		# -----------------------
		six_box = RoundedRectangle(color=GREEN, 
			stroke_width=4, 
			height=5.8, width=1.2, 
			corner_radius=0.2).move_to(digits[6]).shift(DOWN*2.15)
		self.play(FadeIn(six_box))


		# Leave nothing but sixes
		self.remove(*digits, *ddd, *last_row)
		self.add(sixes_group)

		self.wait(2)

		# Transition to 6 stack
		self.remove(six_box)

		scale = 0.05
		stack_h = 10

		six_l = [None] * stack_h # Normal square
		six = [None] * stack_h # Stack
		six_f = [None] * stack_h # Flattened stack

		digit_choice = 6

		count = 0
		for i in range(0,stack_h):
			j = digit_choice

			# Normal version
			bg_img_np = imgToMtx("casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png")
			mtx_l = colorMatrix(bg_img_np, no_text=True).mtx()
			mtx_l.scale(scale*1.5)
			mtx_l.shift(UP*i*0.7)
			
			# Stack version
			mtx = colorMatrixSkew(bg_img_np, shadow=True).mtx()
			mtx.scale(scale)
			mtx.shift(UP*i*0.3)

			# Flat version
			dim = bg_img_np.shape[0]
			bg_img_np_flat = bg_img_np.reshape((dim*dim,1))
			mtx_f = colorMatrixSkew(bg_img_np_flat, shadow=True).mtx()
			mtx_f.scale(scale)
			mtx_f.shift(UP*i*0.3)

			six_l[i] = mtx_l
			six[i] = mtx
			six_f[i] = mtx_f

		six_list = VGroup(*six_l)
		six_list.shift(DOWN*3)
		
		stack = VGroup(*six)
		stack.shift(DOWN*1.5)
		
		stack_f = VGroup(*six_f)
		stack_f.shift(DOWN*1.5)
		
		self.play(ApplyMethod(sixes_group.move_to, ORIGIN))
		self.remove(sixes_group)
		self.play(FadeIn(six_list))
		self.wait()

class SixStack(Scene):
	def construct(self):

		scale = 0.05
		stack_h = 10

		six_l = [None] * stack_h # Normal square
		six = [None] * stack_h # Stack
		six_f = [None] * stack_h # Flattened stack

		digit_choice = 6

		count = 0
		for i in range(0,stack_h):
			j = digit_choice

			# Normal version
			bg_img_np = imgToMtx("casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png")
			mtx_l = ColorMatrix(bg_img_np, no_text=True).mtx()
			mtx_l.scale(scale*1.5)
			mtx_l.shift(UP*i*0.7)
			
			# Stack version
			mtx = ColorMatrixSkew(bg_img_np, shadow=True).mtx()
			mtx.scale(scale)
			mtx.shift(UP*i*0.3)

			# Flat version
			dim = bg_img_np.shape[0]
			bg_img_np_flat = bg_img_np.reshape((dim*dim,1))
			mtx_f = ColorMatrixSkew(bg_img_np_flat, shadow=True).mtx()
			mtx_f.scale(scale)
			mtx_f.shift(UP*i*0.3)

			six_l[i] = mtx_l
			six[i] = mtx
			six_f[i] = mtx_f

		six_list = VGroup(*six_l)
		six_list.shift(DOWN*3)
		
		stack = VGroup(*six)
		stack.shift(DOWN*1.5)
		
		stack_f = VGroup(*six_f)
		stack_f.shift(DOWN*1.5)
		
		self.add(six_list)
		self.wait(1)
		self.play(ReplacementTransform(six_list, stack))
		self.wait(1)
		self.play(ReplacementTransform(stack, stack_f))
		self.wait()

		self.play(ApplyMethod(stack_f.shift,3.5*LEFT))
		self.wait()

		cov_arrow = Arrow([-1.7,0,0],[0.5,0,0], color=WHITE, stroke_width=5)
    
		cov_np = np.array(cov_mtx_6) # imported from cov_mtx_6.py for keeping things clean
		cov_mtx = ColorMatrix(cov_np, no_text=True, norm_colors=True).mtx().shift(RIGHT*3)
		cov_mtx.scale(0.07)

		cov_label = TextMobject("64 $\\times$ 64\\\covariance matrix").shift(RIGHT*3)
		cov_label.scale(1)
		cov_group = VGroup(cov_mtx, cov_label)
		self.play(FadeInFrom(cov_arrow, LEFT))
		self.add(cov_group)

		self.wait()

		self.remove(stack_f, cov_arrow)
		self.play(ApplyMethod(cov_group.shift,6*LEFT))
		# restore line above when rendering final - this is just for speed
		# cov_group.shift(6*LEFT)
		self.wait()

		## Get eigenvectors
		eig_arrow = Arrow([-0.6,0,0],[0.6,0,0], color=WHITE, stroke_width=5)

		eig_np = np.array(eig_mtx_6) # imported from cov_mtx_6.py for keeping things clean
		eig_mtx = ColorMatrix(eig_np, no_text=True, norm_colors=True).mtx().shift(RIGHT*3)
		eig_mtx.scale(0.07)

		eig_label = TextMobject("64 unit eigenvectors").shift(RIGHT*3+UP*0.2)
		eig_label2 = TextMobject("64D each").shift(RIGHT*3+DOWN*0.4)
		eig_label.scale(0.9)
		eig_label2.scale(0.9)
		eig_group = VGroup(eig_mtx, eig_label, eig_label2)
		self.play(FadeInFrom(eig_arrow, LEFT))
		self.add(eig_group)

		self.wait()

		self.remove(eig_arrow, cov_group)
		self.wait()
		self.play(ApplyMethod(eig_group.move_to,ORIGIN))

		self.wait()

		# Take top N eigenvectors
		top_five_np = eig_np[:,0:5]

class GetTopEig(Scene):
	def construct(self):
		eig_np = np.array(eig_mtx_6) # imported from cov_mtx_6.py for keeping things clean
		eig_mtx = colorMatrix(eig_np, no_text=True, norm_colors=True).mtx()
		eig_mtx.scale(0.07)

		eig_label = TextMobject("64 unit eigenvectors").shift(UP*0.2)
		eig_label2 = TextMobject("64D each").shift(DOWN*0.4)
		eig_label.scale(0.9)
		eig_label2.scale(0.9)
		eig_group = VGroup(eig_mtx, eig_label, eig_label2).move_to(ORIGIN)

		self.add(eig_group)

		# Take top 5 eigenvectors
		e = [None] * 5
		for i in range(5):
			e[i] = TextMobject("$\\lambda_"+str(i+1)+"$", color=BRIGHT_RED).shift(RIGHT*i*0.1+UP*2.5+LEFT*2.3)
			e[i].scale(0.8)

		# a = eig_mtx[0].get_corner(TOP+LEFT)
		# b = eig_mtx[63].get_corner(TOP+RIGHT)
		# c = eig_mtx[64*64].get_corner(BOTTOM+RIGHT)
		# d = eig_mtx[64*63].get_corner(BOTTOM+LEFT)

		# a = eig_mtx[0].get_corner(TOP+LEFT)
		# b = eig_mtx[30].get_corner(TOP+RIGHT)
		# c = eig_mtx[64*63+30].get_corner(BOTTOM+RIGHT)
		# d = eig_mtx[64*64].get_corner(BOTTOM+LEFT)

		overlay = [None]*5

		for i in range(5):
			a = eig_mtx[i].get_corner(TOP+LEFT)
			b = eig_mtx[i].get_corner(TOP+RIGHT)
			c = eig_mtx[64*63+i].get_corner(BOTTOM+RIGHT)
			d = eig_mtx[64*63+i].get_corner(BOTTOM+LEFT)

			overlay[i] = Polygon(a,b,c,d, fill_color=BRIGHT_RED, fill_opacity=0.5, stroke_width=0)
			overlay2 = Polygon(a,b,c,d, fill_color=BRIGHT_RED, fill_opacity=0.9, stroke_width=0)

			self.add(overlay2, e[i])
			self.wait(0.7)
			self.remove(e[i], overlay2)
			self.add(overlay[i])
		self.wait()

		self.remove(eig_group)
		self.remove(*overlay)
		top_eig_np = eig_np[:,0:5]
		new_pos = (eig_group.get_corner(TOP+LEFT)+eig_group.get_corner(BOTTOM+LEFT))/2 + RIGHT*0.18
		top_eig_mtx = colorMatrix(top_eig_np, no_text=True, norm_colors=True).mtx().move_to(new_pos)
		top_eig_mtx.scale(0.07)
		a = top_eig_mtx.get_corner(TOP+LEFT)
		b = top_eig_mtx.get_corner(TOP+RIGHT)
		c = top_eig_mtx.get_corner(BOTTOM+RIGHT)
		d = top_eig_mtx.get_corner(BOTTOM+LEFT)
		top_eig_overlay = Polygon(a,b,c,d, fill_color=BRIGHT_RED, fill_opacity=0.5, stroke_width=0)
		top_eig_group = VGroup(top_eig_mtx, top_eig_overlay)
		self.add(top_eig_group)
		self.play(ApplyMethod(top_eig_group.move_to,ORIGIN))
		new_basis_text = TextMobject("New basis:", color=BRIGHT_RED).shift(RIGHT*1.6)
		new_basis_text2 = TextMobject("5 unit eigenvectors,", color=BRIGHT_RED).shift(RIGHT*2.55,DOWN*0.6)
		new_basis_text3 = TextMobject("64D each", color=BRIGHT_RED).shift(RIGHT*1.4,DOWN*1.1)
		n_b_group = VGroup(new_basis_text, new_basis_text2, new_basis_text3)
		n_b_group.shift(RIGHT*0.1)
		self.add(n_b_group)
		self.wait()

class PrincipalComponents(Scene):
	def construct(self):
		k = 5

		eig_np = np.array(eig_mtx_6) # imported from cov_mtx_6.py for keeping things clean
		top_eig_np = eig_np[:,0:k]
		top_eig_mtx1 = colorMatrix(top_eig_np, no_text=True, norm_colors=True).mtx().move_to(ORIGIN)
		top_eig_mtx2 = colorMatrix(top_eig_np, no_text=True, norm_colors=True).mtx().move_to(ORIGIN)
		top_eig_mtx1.scale(0.07)
		top_eig_mtx2.scale(0.07)
		top_eig_mtx = top_eig_mtx1
		a = top_eig_mtx.get_corner(TOP+LEFT)
		b = top_eig_mtx.get_corner(TOP+RIGHT)
		c = top_eig_mtx.get_corner(BOTTOM+RIGHT)
		d = top_eig_mtx.get_corner(BOTTOM+LEFT)
		top_eig_overlay = Polygon(a,b,c,d, fill_color=BRIGHT_RED, fill_opacity=0.5, stroke_width=0)
		top_eig_group = VGroup(top_eig_mtx, top_eig_overlay)
		self.add(top_eig_group)
		self.wait()

		

		e = [None]*k
		e2 = [None]*k
		f = [None]*k
		f2 = [None]*k
		shape = eig_np.shape[0]
		for i in range(k):
			col_list1 = [None] * 64
			col_list2 = [None] * 64
			for j in range(64):
				col_list1[j] = top_eig_mtx1[0+(j)*k+i]
				col_list2[j] = top_eig_mtx2[0+(j)*k+i]
				print("Putting idx {} in col {}".format(0+(j)*k, i))
			e[i] = VGroup(*col_list1)
			e2[i] = VGroup(*col_list2)
			
			a = e[i].get_corner(TOP+LEFT)
			b = e[i].get_corner(TOP+RIGHT)
			c = e[i].get_corner(BOTTOM+RIGHT)
			d = e[i].get_corner(BOTTOM+LEFT)

			a2 = e2[i].get_corner(TOP+LEFT)
			b2 = e2[i].get_corner(TOP+RIGHT)
			c2 = e2[i].get_corner(BOTTOM+RIGHT)
			d2 = e2[i].get_corner(BOTTOM+LEFT)
			
			overlay = Polygon(a,b,c,d, fill_color=BRIGHT_RED, fill_opacity=0.5, stroke_width=0)
			overlay2 = Polygon(a2,b2,c2,d2, fill_color=BRIGHT_RED, fill_opacity=0.5, stroke_width=0)
			f[i] = VGroup(e[i], overlay)
			f2[i] = VGroup(e2[i], overlay2)
			f2[i].shift(RIGHT*(i-(k-1)/2))

		togeth = VGroup(*f2)
		split_apart = VGroup(*f)

		pc_text = TextMobject("Principal Components", color=BRIGHT_RED).move_to(TOP).shift(DOWN)
		
		self.wait()
		self.remove(top_eig_group)
		self.play(Transform(f[0], f2[0]), 
			Transform(f[1], f2[1]), 
			Transform(f[2], f2[2]), 
			Transform(f[3], f2[3]), 
			Transform(f[4], f2[4]))

		self.wait()
		self.add(pc_text)
		self.wait()
			





class TestMatrixSkew(Scene):
	def construct(self):
		basis = RoundedRectangle(height=3, width=0.7, corner_radius=0.2).move_to(DOWN*0.81+LEFT*1.65)

		# requires list of strings
		matrix = Matrix([["1","2"],["3","4"]]).shift(2*RIGHT)

		self.add(matrix)
		self.wait(1)
		j = 6
		i = 0
		bg_img = "casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png"
		
		mtx = imgToMtx(bg_img)
		print("Returned mtx from image.")

		#mtx = np.asarray([[1,0,1],[0.5,1,2],[3,4,5]])
		#mtx_flat = np.asarray([[1,0,1],[0.5,1,2],[3,4,5]]).reshape(9,1)
		
		matrix_group = colorMatrix(mtx, no_text=True, tint='R').mtx().shift(2*LEFT)
		matrix_group.scale(0.2)

		matrix_group_skew = colorMatrixSkew(mtx).mtx().shift(2*LEFT)
		matrix_group_skew.scale(0.1)
		
		matrix_group2 = colorMatrix(mtx, no_text=True).mtx().shift(2*LEFT)
		matrix_group2.scale(0.2)


		self.add(matrix_group)
		self.wait(1)
		
		self.play(ReplacementTransform(matrix_group, matrix_group_skew))
		self.wait()
		self.play(Transform(matrix_group_skew, matrix_group2))
		
		self.wait()

class TestVGroup(Scene):
	def construct(self):
		rect1 = Rectangle(height=1, width=1).set_color(WHITE)
		rect2 = Rectangle(height=1, width=1).shift(LEFT)
		rect3 = rect2
		rect3.set_color(RED)

		group = VGroup(rect1, rect2)



		self.add(group)
		self.wait()
		self.play(ApplyMethod(group[0].shift,5*LEFT))
		self.play(Transform(group[0], rect3))

		self.wait()

class Square_to_vec(Scene):
	# python extract_scene.py casey_sandbox/manim_tutorial_1.py Square_to_vec -pl -g
	# convert -delay 7 -loop 0 *.png myimage.gif
	# convert myimage.gif -coalesce -repage 0x0 -crop 550x100+120+190 +repage output.gif
	def construct(self):

		text = TextMobject("X = ", color=WHITE).move_to(LEFT*3.9).shift(UP*0.02)

		j = 2
		i = 7
		bg_img = "casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png"
		mtx = imgToMtx(bg_img)

		mtx_1 = mtx.reshape((1,-1))

		matrix_group = ColorMatrix(mtx, no_text=True, tint='None').mtx()
		matrix_group.scale(0.1)

		# matrix_group2 = matrix_group

		self.add(matrix_group)
		self.wait(1)

		# matrix_group_skew = ColorMatrixSkew(mtx).mtx().shift(2*LEFT)
		# matrix_group_skew.scale(0.1)

		matrix_group2 = ColorMatrix(mtx_1, no_text=True, tint='None').mtx()
		matrix_group2.scale(0.1)

		self.play(ReplacementTransform(matrix_group, matrix_group2))
		self.add(text)
		self.wait(3)

class TwoStack(Scene):
	# python extract_scene.py casey_sandbox/manim_tutorial_1.py TwoStack -pl -g
	# convert -delay 7 -loop 0 *.png myimage.gif
	# convert myimage.gif -coalesce -repage 0x0 -crop 420x440+175+10 +repage output.gif

	def construct(self):

		scale = 0.05
		stack_h = 10

		six_l = [None] * stack_h # Normal square
		six = [None] * stack_h # Stack
		six_f = [None] * stack_h # Flattened stack

		digit_choice = 2

		count = 0
		for i in range(0,stack_h):
			j = digit_choice

			# Normal version
			bg_img_np = imgToMtx("casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png")
			mtx_l = ColorMatrix(bg_img_np, no_text=True).mtx()
			mtx_l.scale(scale*1.5)
			mtx_l.shift(UP*i*0.7)
			
			# Stack version
			mtx = ColorMatrixSkew(bg_img_np, shadow=True).mtx()
			mtx.scale(scale)
			mtx.shift(UP*i*0.3)

			# Flat version
			dim = bg_img_np.shape[0]
			bg_img_np_flat = bg_img_np.reshape((dim*dim,1))
			mtx_f = ColorMatrixSkew(bg_img_np_flat, shadow=True).mtx()
			mtx_f.scale(scale)
			mtx_f.shift(UP*i*0.3)

			six_l[i] = mtx_l
			six[i] = mtx
			six_f[i] = mtx_f

		six_list = VGroup(*six_l)
		six_list.shift(DOWN*3)
		
		stack = VGroup(*six)
		stack.shift(DOWN*1.5)
		
		stack_f = VGroup(*six_f)
		stack_f.shift(DOWN*1.5)
		
		self.add(six_list)
		self.wait(1)
		self.play(ReplacementTransform(six_list, stack))
		self.wait(1)
		self.play(ReplacementTransform(stack, stack_f))
		self.wait()

		self.play(ApplyMethod(stack_f.shift,2*LEFT))
		self.wait()


		D_sq = DoubleArrow([-3.6,3.5,0],[0.0,0,0], color=YELLOW)
		D_sq_text = TextMobject("$D^2$: number of pixels", color=YELLOW).move_to(0).shift(RIGHT*0.7+UP*1.8)
		D_sq_text.scale(0.8)
		
		D_group = VGroup(D_sq, D_sq_text)
		self.play(FadeIn(D_group))
		self.wait()
		self.play(FadeOut(D_group))

		N = DoubleArrow([0.2,-0.3,0],[0.2,-3.3,0], color=YELLOW)
		N_text = TextMobject("$N$: number of images", color=YELLOW).move_to(0).shift(RIGHT*2.5+DOWN*2)
		N_text.scale(0.8)
		
		N_group = VGroup(N, N_text)
		self.play(FadeIn(N_group))
		self.wait()
		self.play(FadeOut(N_group))
		self.wait()

class Two_stack_collapse(Scene):
	# python extract_scene.py casey_sandbox/manim_tutorial_1.py Two_stack_collapse -pl -g

	# conversion should speed up alot

	# convert -delay 7 -loop 0 *.png myimage.gif
	# convert myimage.gif -coalesce -repage 0x0 -crop 390x475+170+0 +repage output.gif

	def construct(self):

		HOF = 1 # height offset

		scale = 0.05
		stack_h = 10

		six_l = [None] * stack_h # Normal square
		six = [None] * stack_h # Stack
		six_f = [None] * stack_h # Flattened stack

		digit_choice = 2

		count = 0
		imgs_flat = []
		for i in range(0,stack_h):
			j = digit_choice

			# Normal version
			bg_img_np = imgToMtx("casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png")
			mtx_l = ColorMatrix(bg_img_np, no_text=True).mtx()
			mtx_l.scale(scale*1.5)
			mtx_l.shift(UP*i*0.7)
			
			# Stack version
			mtx = ColorMatrixSkew(bg_img_np, shadow=False).mtx()
			mtx.scale(scale)
			mtx.shift(UP*i*0.3)

			# Flat version
			dim = bg_img_np.shape[0]
			bg_img_np_flat = bg_img_np.reshape((dim*dim,1))
			imgs_flat.append(bg_img_np_flat)
			
			mtx_f = ColorMatrixSkew(bg_img_np_flat, shadow=False).mtx()
			mtx_f.scale(scale)
			mtx_f.shift(UP*i*0.3)

			six_l[i] = mtx_l
			six[i] = mtx
			six_f[i] = mtx_f

		six_list = VGroup(*six_l)
		six_list.shift(DOWN*3)
		
		stack = VGroup(*six)
		stack.shift(DOWN*1.5)
		
		stack_f = VGroup(*six_f)
		stack_f.shift(DOWN*HOF)
		
		# self.add(stack_f)
		# self.wait()

		pixel_col = [None] * (dim*dim)
		pixel_col_groups = [None] * (dim*dim)

		for j in range(0, dim*dim):
			pixel_col[j] = []
			for i in range(0, stack_h):
				pixel_col[j].append(six_f[i][j])

			pixel_col_groups[j] = VGroup(*pixel_col[j])

		all_group = VGroup(*pixel_col_groups)
		self.add(all_group)

		cb_text = TextMobject("$\\mathbf{X} =$")
		cb_text.shift(UP*(3-HOF)+LEFT*3.3)

		cb = CurlyBrace().brace()
		cb.scale(0.5)
		cb.rescale_to_fit(3,1,stretch=True)
		cb.shift(UP*(1.5+2.7-HOF)+LEFT*2.2)
		
		cb_group = VGroup(cb_text, cb)
		
		self.add(cb_group)
		self.wait(1)

		means = np.mean(imgs_flat, axis=0)
		mean_skew = ColorMatrixSkew(means, shadow=False).mtx()
		mean_skew.scale(scale)
		mean_skew.shift(DOWN*(1+HOF))


		# for j in range(0,1):
		# for j in range(dim*dim-1,dim*dim):
		#for j in range(0, dim*dim):
		mu_text = TextMobject("$\\mu_X =$")
		mu_text.shift(DOWN*(1.5-HOF)+LEFT*2.5)
		self.play(FadeOut(cb_group))
		self.play(FadeIn(mu_text), Transform(all_group, mean_skew))
		
		
		self.wait(2)

class mu_one(Scene):
	# python extract_scene.py casey_sandbox/manim_tutorial_1.py Two_stack_collapse -pl -g


	# convert -delay 7 -loop 0 *.png myimage.gif
	# convert myimage.gif -coalesce -repage 0x0 -crop 420x440+175+10 +repage output.gif

	def construct(self):

		HOF = 1 # height offset

		scale = 0.05
		stack_h = 10

		six_l = [None] * stack_h # Normal square
		six = [None] * stack_h # Stack
		six_f = [None] * stack_h # Flattened stack

		digit_choice = 2

		count = 0
		imgs_flat = []
		for i in range(0,stack_h):
			j = digit_choice

			# Normal version
			bg_img_np = imgToMtx("casey_sandbox/scenes/1_intro/assets/tiny/" + str(j) + ".0_" + str(i) +".png")
			mtx_l = ColorMatrix(bg_img_np, no_text=True).mtx()
			mtx_l.scale(scale*1.5)
			mtx_l.shift(UP*i*0.7)
			
			# Stack version
			mtx = ColorMatrixSkew(bg_img_np, shadow=False).mtx()
			mtx.scale(scale)
			mtx.shift(UP*i*0.3)

			# Flat version
			dim = bg_img_np.shape[0]
			bg_img_np_flat = bg_img_np.reshape((dim*dim,1))
			imgs_flat.append(bg_img_np_flat)
			
			mtx_f = ColorMatrixSkew(bg_img_np_flat, shadow=False).mtx()
			mtx_f.scale(scale)
			mtx_f.shift(UP*i*0.3)

			six_l[i] = mtx_l
			six[i] = mtx
			six_f[i] = mtx_f

		six_list = VGroup(*six_l)
		six_list.shift(DOWN*3)
		
		stack = VGroup(*six)
		stack.shift(DOWN*1.5)
		
		stack_f = VGroup(*six_f)
		stack_f.shift(DOWN*HOF)
		
		# self.add(stack_f)
		# self.wait()

		pixel_col = [None] * (dim*dim)
		pixel_col_groups = [None] * (dim*dim)

		for j in range(0, dim*dim):
			pixel_col[j] = []
			for i in range(0, stack_h):
				pixel_col[j].append(six_f[i][j])

			pixel_col_groups[j] = VGroup(*pixel_col[j])

		self.add(*pixel_col_groups)

		cb_text = TextMobject("$\\mathbf{X} =$")
		cb_text.shift(UP*(3-HOF)+LEFT*3.3)

		cb = CurlyBrace().brace()
		cb.scale(0.5)
		cb.rescale_to_fit(3,1,stretch=True)
		cb.shift(UP*(1.5+2.7-HOF)+LEFT*2.2)
		
		cb_group = VGroup(cb_text, cb)
		
		self.add(cb_group)
		self.wait(1)

		means = np.mean(imgs_flat, axis=0)
		mean_skew = ColorMatrixSkew(means, shadow=False).mtx()
		mean_skew.scale(scale)
		mean_skew.shift(DOWN*(1+HOF))


		
		# self.play(FadeOut(cb_group))

		for j in range(0,dim*dim):
			if j >= 3 and j <= 62:
				continue

			mu_text = TextMobject("$E\\left[X_{%s}\\right]$" % str(j+1))
			
			if j == 63:
				arrow = Arrow(DOWN*(4-HOF)+RIGHT*0.3, DOWN*3.6+RIGHT*1.5, color=WHITE)
				arrow.shift(offset)
				mu_text.shift(DOWN*(4-HOF)+LEFT*0.5)
				mu_text.shift(offset)
			else:
				offset = (0.07*j) * RIGHT + (0.03*j) * DOWN
				
				arrow = Arrow(DOWN*(2.3-HOF)+LEFT*2.4, DOWN*0.4+LEFT*1.45, color=WHITE)
				arrow.shift(offset)

				mu_text.shift(DOWN*(2.5-HOF)+LEFT*2.5)
				mu_text.shift(offset)
			mu_group = VGroup(mu_text, arrow)

			goback = pixel_col_groups[j].deepcopy()

			self.play(ReplacementTransform(pixel_col_groups[j], mean_skew[j]))
			self.play(FadeIn(mu_group))
			self.wait(1)
			self.play(FadeOut(mu_group), ReplacementTransform(mean_skew[j], goback))
			self.wait(1)
		
		self.wait(2)
		



