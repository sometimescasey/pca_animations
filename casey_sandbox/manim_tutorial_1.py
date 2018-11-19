from big_ol_pile_of_manim_imports import *
from once_useful_constructs.matrix_multiplication import *
from casey_sandbox.colorMatrix import *

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

# def colorMatrix():
# 	mtx = np.asarray([[1,0],[0.5,1]])
# 	m = mtx.shape[0]
# 	n = mtx.shape[1]

# 	s = 1.0 # width of each square

# 	sq_list = []
# 	sq_num = []

# 	count=0
# 	for r in range(m):
# 		for c in range(n):
# 			shade = (mtx[r][c])
# 			sq = Square(side_length=1.0, fill_opacity=1, stroke_width=0, fill_color=Color(rgb=(shade, shade, shade))).move_to(r*UP+c*LEFT)
# 			sq_list.append(sq)
# 			label = TextMobject(str(mtx[r][c]),color=Color(rgb=(1,0,0)),size=8,background_stroke_width=0).move_to(r*UP+c*LEFT)
# 			sq_num.append(label)
# 			# self.add(sq_list[count], sq_num[count])
# 			count += 1

# 	outline = Rectangle(height=s*m, width=s*n).move_to(s*UP/m+s*LEFT/n)

# 	# * operator expands lists before the function call
# 	group = VGroup(outline, *sq_list, *sq_num)
# 	# self.add(outline, *sq_list, *sq_num)
# 	return group

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
		matrix_group = colorMatrix(mtx, width=0.5, text_scale=0.5).shift(2*LEFT)
		matrix_group2 = colorMatrix(mtx_flat, width=0.5, text_scale=0.5).shift(4*LEFT)
		matrix_group3 = colorMatrix(mtx, width=0.5, text_scale=0.5).shift(2*LEFT)

		self.add(matrix_group)
		self.wait(1)
		self.play(ReplacementTransform(matrix_group, matrix_group2))
		self.wait()
		self.play(Transform(matrix_group2, matrix_group3))
		self.wait()

class Intro(Scene):
	def construct(self):
		pca = TextMobject("Principal Component Analysis (PCA)").move_to(TOP+DOWN*1)
		pca.scale(1)

		one = ImageMobject("casey_sandbox/pup.jpg", invert=False, image_mode="RGBA")
		one.scale(1)
		self.add(one, pca)
		self.wait(2)
