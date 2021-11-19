
#Import explainer 
from tf_explain.core.grad_cam import GradCAM
#Instantiation of the explainer
explainer = GradCAM()
# Call to explain() method
output = explainer.explain(x_batch_test[0],model,np.argmax(y_batch_test[0]))
# Save output
#explainer.save(output, output_dir, output_name)