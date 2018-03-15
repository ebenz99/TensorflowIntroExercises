import tensorflow as tf

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')	#y is dependent on x	

with tf.Session() as session:
    merged = tf.summary.merge_all()		#summarizes connections into nodes
    writer = tf.summary.FileWriter("C:\\Code\\python\\tensorflowLesson1\\TensorflowIntroExercizes", session.graph)	#saves graph of connections
    model =  tf.global_variables_initializer()			#shows y connection
    session.run(model)		
    print(session.run(y))		#gives value of y from original node

#after this to view at web address given, run command:   tensorboard --logdir=mylogs:C:\Code\python\tensorflowLesson1\TensorflowIntroExercizes
	#mylogs is a name to give to get rid of colon ambiguity