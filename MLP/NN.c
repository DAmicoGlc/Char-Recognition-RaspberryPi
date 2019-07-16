/**
* @file NN.c
* @author Gianluca D'Amico
* @brief Stand alone MLP training 
*
* HANDLING MLP TRAINING AND TESTING
*/

#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <allegro.h>

/**
* LOCAL CONSTANTS
*/

#define USAGE "ERROR, Usage: h n...n i b l e m \n\
-h = (# of hidden layer) \n\
-n...n = (# of neuron in first hidden layer)...(# of neuron in last hidden layer) \n\
-i = (Training iteration, e.g. Epoches) \n\
-b = (Size of mini batch to compute sthocastic gradient) \n\
-l = (Learning rate of training phase) \n\
-e = (epsilon of Error in training phase) \n\
-m = (momentum of training phase) \n\
All parameter must be grather than 0. The learning rate must be also less than 1.\n"

/**< Set type. */
#define EXAMPLE_TYPE "balanced"

/**< Training set file. */
#define TRAIN_SET_IMAGE "data/emnist-balanced-train-images-idx3-ubyte"
#define TRAIN_SET_LABEL "data/emnist-balanced-train-labels-idx1-ubyte"

/**< Testing set file. */
#define TEST_SET_IMAGE "data/emnist-balanced-test-images-idx3-ubyte"
#define TEST_SET_LABEL "data/emnist-balanced-test-labels-idx1-ubyte"

/**< Training set cardinality. */
#define TRAIN_NUMBER 112800
#define TRAIN_LOOP 101520
#define VALIDATION_LOOP 11280

/**< Cardinality for each class. */
#define TOT_PER_CLASS 2400
#define TRA_PER_CLASS 2160
#define VAL_PER_CLASS 240

/**< Teesting set cardinality. */
#define TEST_NUMBER 1000

/**< Number of class. */
#define OUTPUT_SIZE 47
/**< Pixel of images. */
#define INPUT_SIZE 784

#define DISPLAY_HEIGHT 520
#define DISPLAY_WIDTH 1060

#define RANDOM_MEAN 0
#define RANDOM_STD_DEVIATION 0.5

#define rando() ((double)rand()/((double)RAND_MAX+1))

/**
* LOCAL STRUCTS
*/

/**< Struct of the sinapsi. */
struct TypeSinapsiMatrix {
	double** weights;
	double* bias;

	double** gradient_weights;
	double* gradient_bias;

	double** prev_weights_variation;
	double* prev_bias_variation;

	int card_in;
	int card_out;
};

/**< Struct of the layer. */
struct TypeLayerPair {
	double* activation_value;
	double* x_value;

	double* delta;

	int num_neuron;
};

/**< Struct of examples. */
struct TypeDataSet {
	int pixel[INPUT_SIZE];
	int label;
};

/**< Type definition. */
typedef struct TypeSinapsiMatrix type_sinapsi;		
typedef struct TypeLayerPair type_layer;		
typedef struct TypeDataSet type_data;

/**< Struct of the sets. */
struct TypeTrainSet {
	type_data* classes[OUTPUT_SIZE][TOT_PER_CLASS];
	int sizes[OUTPUT_SIZE];
};

/**< Type definition. */
typedef struct TypeTrainSet type_train_set;

/**
* GLOBAL VARIABLE
*/

/**< Training set. */
type_train_set train_set;

/**< First erro to set the scale of the graphs. */
double max_err;
double val_max_acc = 0;

/**< Earling stop offset. */
double val_offset = 0.3;

/**
* FUNCTION PROTOTYPES
*/

/**< init layer and sinapsi. */
type_layer* init_layer(int,int);
type_sinapsi* init_sinapsi(int,int);

/**< forward propagation of result. */
void propagate_into_layer(type_layer*,type_layer*,type_sinapsi*,int);

/**< compute neuron delta of the output layer and the sthocastic gradient. */
void compute_output_delta(int*,type_layer*,type_sinapsi*,type_layer*,double);

/**< compute neuron delta of others layer and the sthocastic gradient. */
void compute_layer_delta(type_sinapsi*,type_sinapsi*,type_layer*,
											type_layer*,type_layer*,double);

/**< update weights and bias at the end of the mini bacth. */
void update_sinapsi(type_sinapsi*,int,double);

/**< transfer function y=f(x)=1/(1-e^(-x)). */
double logistic_function(double);
/**< derivate of transfer function y=f(x)=1/(1-e^(-x)). */
double derivate_logistic_function(double);

/**< UTILS */
double get_rand(int,int);
/**< Gaussian distribution generation */
void generate_gaussian_random(double*,int);
/**< Set shuffling */
void shuffle(type_data**,int);
/**< Pick example for validation and training set */
void pick_example(type_data**,type_data**);
/**< Read files */
int read_train_set(FILE*,FILE*);
int read_test_set(FILE*,FILE*,type_data**);
/**< Save reslting weights and bias */
void save_file(type_sinapsi**,int,double,char*);

/**< ALLEGRO UTILS */
void draw_point(double,int,double,int);
void init_display();
void draw_image (type_data*);

/**
* @brief Initilize Layer.
*
* Allocate the memory for the structure of the layer.
*
* @param cardinality is the number of neurons
* @param input if 1 is th einput layer
* @return layer struct pointer
*/
type_layer* init_layer(int cardinality, int input) {

	type_layer* layer = (type_layer*) malloc(sizeof(type_layer));

	layer->activation_value = (double*) malloc(cardinality*sizeof(double));
	layer->x_value = (double*) malloc(cardinality*sizeof(double));
	
	/**< if it is not the layer input i will allocate memory for delta*/
	if (input>0) 
		layer->delta = (double*) malloc(cardinality*sizeof(double));
	

	layer->num_neuron = cardinality;

	return layer;
}

/**
* @brief Initilize Sinapsi.
*
* Allocate the memory for the structure of the sinapsi.
*
* @param card_in is the number of neurons in the input layer
* @param card_out is the number of neurons in the output layer
* @return sinapsi struct pointer
*/
type_sinapsi* init_sinapsi(int card_in, int card_out) {

	int i, j, count=0;
	int dim = card_in*card_out + card_in + (card_in*card_out + card_in)%2;
	double* gaussian_seq;

	gaussian_seq = (double*) calloc(dim,sizeof(double));

	type_sinapsi* sinapsi = (type_sinapsi*) malloc(sizeof(type_sinapsi));

	/**< create matrix of weights*/
	sinapsi->weights = (double**) malloc(sizeof(double*)*card_out);				

	for (i = 0; i < card_out; ++i)
		sinapsi->weights[i] = (double*) malloc(sizeof(double)*card_in);


	/**<  create array of bias*/
	sinapsi->bias = (double*) malloc(sizeof(double)*card_out);

	/**<  create matrix of sthocastic gredient of weights*/
	sinapsi->gradient_weights = (double**) malloc(sizeof(double*)*card_out);

	for (i = 0; i < card_out; ++i)
		sinapsi->gradient_weights[i] = (double*) calloc(card_in,sizeof(double));

	/**<  create array of sthocastic gredient of bias*/
	sinapsi->gradient_bias = (double*) calloc(card_out,sizeof(double));

	/**<  create matrix of sthocastic gredient of weights*/
	sinapsi->prev_weights_variation = (double**) 
												malloc(sizeof(double*)*card_out);		
	for (i = 0; i < card_out; ++i)
		sinapsi->prev_weights_variation[i] = (double*) 
												calloc(card_in,sizeof(double)); 

	/**<  create array of sthocastic gredient of bias*/
	sinapsi->prev_bias_variation = (double*) calloc(card_out,sizeof(double));

	sinapsi->card_in = card_in;
	sinapsi->card_out = card_out;

	/**< Initilize the weights values*/
	generate_gaussian_random(gaussian_seq, dim);

	for (i = 0; i < card_out; ++i) {

		for (j = 0; j < card_in; ++j) {
			sinapsi->weights[i][j] = gaussian_seq[count];
			count++;
		}

		sinapsi->bias[i] = gaussian_seq[count];
		count++;
	}

	return sinapsi;
}

/**
* @brief Sigmoid function.
*
* @param x value to compute
* @return resulting value
*/
double logistic_function(double x) {
    return 1 / (1 + exp(-x));
}

/**
* @brief Derivative of the Sigmoid function.
*
* @param x value to compute
* @return resulting value
*/
double derivate_logistic_function(double x) {
    return logistic_function(x) * ( 1 - logistic_function(x) );
}

/**
* @brief Softmax function.
*
* @param value to compute
* @param sum sum of all exp values
* @param max highest probability value
* @return resulting value
*/
double softmax(double value, double sum, double max) {
	return (exp(value + max) / (sum));
}

/**
* @brief Feedfoward phase
*
* Propagate the result in from the in layer to the out layer given as input
*
* @param prev_layer is previous layer
* @param curr_layer is the current layer
* @param sinapsi is the sinapsi between the layers
* @param flag_out is 1 if the current layer is the output
*/
void propagate_into_layer(type_layer* prev_layer, type_layer* curr_layer, 
										type_sinapsi* sinapsi, int flag_out) {

	int i,j;
	double sum_up;
	int card_in = prev_layer->num_neuron;
	int card_out = curr_layer->num_neuron;

	double** in_weights = sinapsi->weights;
	double* in_bias = sinapsi->bias;

	/**< for each neuron in the curr layer i have to compute the sum*/
	for (i = 0; i < card_out; ++i) {			

		sum_up=0;
		
		/**< the sinapsi is a matix in which the i-th row represent all the */
		/* weight for the i-th output*/
		for (j = 0; j < card_in; ++j) 
			sum_up += in_weights[i][j] * prev_layer->x_value[j];	
		
		sum_up += in_bias[i];

		curr_layer->activation_value[i] = sum_up;
		if (!flag_out)
			curr_layer->x_value[i] = logistic_function(sum_up);
	}

	return;
}

/**
* @brief Back propagation start phase
*
* Compute the delta and the corrispondent gradients of the output layer
*
* @param desired_output is the array of the expected value
* @param output_layer 
* @param sinapsi
* @param prev_layer is the  last layer befor the output one
* @param eta is the learning rate
*/
void compute_output_delta(int* desired_output, type_layer* output_layer, 
				type_sinapsi* sinapsi, type_layer* prev_layer, double eta) {

	int i,j;
	int curr_card = output_layer->num_neuron;
	int prev_card = prev_layer->num_neuron;

	double* delta = output_layer->delta;
	
	double* output_activation = output_layer->activation_value;
	double* output_x = output_layer->x_value;

	double* prev_x = prev_layer->x_value;

	double** gradient_weights = sinapsi->gradient_weights;
	double* gradient_bias = sinapsi->gradient_bias;

	/**< derivate of 1/2 ( a - y )^2 = ( a - y ) * derivate of 
	 * logistic function*/
	for (i = 0; i < curr_card; ++i) {

		delta[i] = (desired_output[i] - output_x[i]);

		/**< update the gradient of weights adding the product of the delta 
		 * with the previous transfer value*/
		for (j = 0; j < prev_card; ++j) {
			sinapsi->gradient_weights[i][j] += eta * delta[i] * prev_x[j];
		}

		/**< update the gradient of bias*/
		sinapsi->gradient_bias[i] += eta*delta[i];
	}
	

	return;
}

/**
* @brief Back propagation
*
* Compute the delta of hte other sinapsi
*
* @param in_sinapsi
* @param out_sinapsi 
* @param prev_layer
* @param curr_layer
* @param next_layer
* @param eta is the learning rate
*/
void compute_layer_delta(type_sinapsi* in_sinapsi, type_sinapsi* out_sinapsi, 
						type_layer* prev_layer, type_layer* curr_layer, 
						type_layer* next_layer, double eta) {

	int i,j;
	double sum_up;
	
	int next_card = next_layer->num_neuron;
	int curr_card = curr_layer->num_neuron;
	int prev_card = prev_layer->num_neuron;

	double** out_weights = out_sinapsi->weights;

	double* next_delta = next_layer->delta;
	double* curr_delta = curr_layer->delta;

	double* curr_activation = curr_layer->activation_value;

	double* prev_x = prev_layer->x_value;

	/**< delta_(l) = ( w_(l+1)T * delta_(l+1) ) * derivate 
	* in ( activation value_(l) ) of trans function
	* gradient_weight_(l-1) = delta_(l) * x_value_(l-1)
	* gradient_weight_(l-1) = delta_(l) */
	for (i = 0; i < curr_card; ++i) {

		sum_up = 0.0;

		/**< the sinapsi is a matix in which the i-th row represent all
		* the weight for the i-th output
		* we need to use the traspose by inverting the index */
		for (j = 0; j < next_card; ++j) 
			sum_up += out_weights[j][i] * next_delta[j];

		curr_delta[i] = sum_up * derivate_logistic_function(curr_activation[i]);

		/**< update the gradient of weights adding the product of the delta 
		 * with the previous transfer value */
		for (j = 0; j < prev_card; ++j) 
			in_sinapsi->gradient_weights[i][j] += eta * 
												curr_delta[i] * prev_x[j];

		/**< update the gradient of bias */
		in_sinapsi->gradient_bias[i] += eta * curr_delta[i];
	}

	return;
}

/**
* @brief Update weights and bias
*
* Update the weights and bias after the backpropagation phase
*
* @param sinapsi
* @param batch_size is the size of the mini-bacth
* @param mi is the momentum
*/
void update_sinapsi(type_sinapsi* sinapsi, int batch_size, double mi) {

	int i,j;

	double temp;

	double** prev_weights_variation = sinapsi->prev_weights_variation;
	double* prev_bias_variation = sinapsi->prev_bias_variation;

	for (i = 0; i < sinapsi->card_out; ++i) {

		for (j = 0; j < sinapsi->card_in; ++j) {
			temp = sinapsi->weights[i][j];
			sinapsi->weights[i][j] += ( 1.0/batch_size ) * 
							sinapsi->gradient_weights[i][j] 
							+ mi * prev_weights_variation[i][j];		
			prev_weights_variation[i][j] = temp - sinapsi->weights[i][j];
		}

		sinapsi->bias[i] += ( 1.0/batch_size ) * 
									sinapsi->gradient_bias[i] 
									+ mi * prev_bias_variation[i];						
		sinapsi->prev_bias_variation[i] = ( 1.0/batch_size ) 
										* sinapsi->gradient_bias[i];
	}	
			
	return;
}

/**
* @brief Read training file
*
* @param fp_label label file descriptor
* @param fp_pixel image file descriptor
*/
int read_train_set(FILE *fp_label, FILE *fp_pixel) {

	int i,j;
	int bytes_read, label;

    unsigned char * buf_label = (unsigned char *) 
								malloc(TRAIN_NUMBER * sizeof(unsigned char));
    unsigned char * buf_pixel = (unsigned char *) 
					malloc(TRAIN_NUMBER * INPUT_SIZE * sizeof(unsigned char));

    bytes_read = fread(buf_label, 1, TRAIN_NUMBER, fp_label);

    if (bytes_read<0)
    	return -1;
    
	bytes_read = fread(buf_pixel, 1, INPUT_SIZE * TRAIN_NUMBER, fp_pixel);

    if (bytes_read<0)
    	return -1;

    /**< read num_bytes bytes at a time and push it in the array*/
    for (i = 0; i < TRAIN_NUMBER; ++i) {

    	if (EXAMPLE_TYPE == "letters")
    		label = (int)buf_label[i] - 1;
    	else	
    		label = (int)buf_label[i];

    	train_set.classes[label][train_set.sizes[label]]->label = label;

    	for (j = 0; j < INPUT_SIZE; ++j) {

	        if ((int)buf_pixel[i*INPUT_SIZE + j] > 0)
	            train_set.classes[label][train_set.sizes[label]]->pixel[j] = 1;
	        else
	            train_set.classes[label][train_set.sizes[label]]->pixel[j] = 0;
		}

		train_set.sizes[label]++;
    }

    free(buf_label);
    free(buf_pixel);

    return 1;
}

/**
* @brief Read testing file
*
* @param fp_label label file descriptor
* @param fp_pixel image file descriptor
* @param data structur in which the set have to be saved
*/
int read_test_set(FILE *fp_label, FILE *fp_pixel, type_data** data) {

	int i,j;
	int bytes_read, label;

    unsigned char * buf_label = (unsigned char *) malloc(TEST_NUMBER * sizeof(unsigned char));
    unsigned char * buf_pixel = (unsigned char *) malloc(INPUT_SIZE * TEST_NUMBER * sizeof(unsigned char));

    bytes_read = fread(buf_label, 1, TEST_NUMBER, fp_label);

    if (bytes_read<0)
    	return -1;
    
	bytes_read = fread(buf_pixel, 1, INPUT_SIZE * TEST_NUMBER, fp_pixel);

    if (bytes_read<0)
    	return -1;

    /**< read num_bytes bytes at a time and push it in the array*/
    for (i = 0; i < TEST_NUMBER; ++i) {

    	if (EXAMPLE_TYPE == "letters")
    		data[i]->label = buf_label[i] - 1;
    	else
    		data[i]->label = buf_label[i];

    	for (j = 0; j < INPUT_SIZE; ++j) {

	        if ((int)buf_pixel[i*INPUT_SIZE + j] > 0)
	            data[i]->pixel[j] = 1;
	        else
	            data[i]->pixel[j] = 0;
	    }
    }

    free(buf_label);
    free(buf_pixel);

    return 1;
}
/**
* @brief Pick example from trainning and validation set
*
* @param train_set_used training set
* @param validation_set validation set
*/
void pick_example(type_data** train_set_used, type_data** validation_set) {

	int i, j;

	for (i = 0; i < OUTPUT_SIZE; ++i) 
		shuffle(train_set.classes[i],train_set.sizes[i]);

	for (i = 0; i < OUTPUT_SIZE; ++i) {
		for (j = 0; j < TRA_PER_CLASS; ++j) 
			train_set_used[i * TRA_PER_CLASS + j] = train_set.classes[i][j];

		for (j = 0; j < VAL_PER_CLASS; ++j)
			validation_set[i * VAL_PER_CLASS + j] = train_set.classes[i][TRA_PER_CLASS + j];
	}

	return;
}

/**
* @brief Save resulting weights and bias
*
* @param sinapsi is the matrix of all sinapsi
* @param num_sinapsi is the number of sinapsi
* @param accuracy is the resulting accuracy
* @param filename
*/
void save_file(type_sinapsi **sinapsi, int num_sinapsi, 
											double accuracy, char* filename) {

	int i,j,k;

	FILE *f = fopen(filename, "w");


	if (f == NULL)
	{
	    printf("Error opening file!\n");
	    exit(1);
	}

	for (k = 0; k < num_sinapsi; ++k)
	{
		for (i = 0; i < sinapsi[k]->card_out; ++i)
		{
			for (j = 0; j < sinapsi[k]->card_in; ++j)
			{
				if (j!=sinapsi[k]->card_in -1)
					fprintf(f, "%2.4f_", sinapsi[k]->weights[i][j]);
				else
					fprintf(f, "%2.4f\n", sinapsi[k]->weights[i][j]);
			}
			fprintf(f, "%2.4f\n", sinapsi[k]->bias[i]);
		}
	}

	fprintf(f,"%2.2f%% of accuracy!\n",accuracy);

	fclose(f);

	return;
}

/**
* @brief Save resulting weights and bias
*
* @param seq is the aray in which the sequence will be saved
* @param dim is the number ofthe number that must be generated
*/
void generate_gaussian_random(double* seq, int dim) {
    int i;
    double x,y,rsq,f;

    for ( i = 0; i < dim; i += 2 )
    {
        do {
            x = 2.0 * rand() / (double)RAND_MAX - 1.0;
            y = 2.0 * rand() / (double)RAND_MAX - 1.0;
            rsq = x * x + y * y;
        }while( rsq >= 1. || rsq == 0. );
        
        f = sqrt( -2.0 * log(rsq) / rsq );
        
        seq[i]   = RANDOM_MEAN + RANDOM_STD_DEVIATION *(x * f);
        seq[i+1] = RANDOM_MEAN + RANDOM_STD_DEVIATION *(y * f);
    }
    
    return;
}

/**
* @brief Generate a random number
*
* @param x is the upper bound
* @param y is the lower bound
*/
double get_rand(int x,int y) {
	double randomi = (rand()/(double)(RAND_MAX))*abs(x-y)-1;
	return randomi;
}

/**
* @brief Shuffel the data set
*
* @param data data set that must be shuffeled
* @param data_amount cardinality of the dataset
*/
void shuffle(type_data **data, int data_amount) {
	int i,j;

	type_data* temp;

    for (i = 0; i < data_amount; i++) {
    	j = get_rand(i,data_amount);

    	temp = data[j];
    	data[j] = data[i];
    	data[i] = temp;
    }

    return; 
}

/**
* @brief Draw a point in the display
*/
void draw_point(double point_y, int point_x, double old_point_y, int flag) {

	int x,y;
	int old_y;	
	char label[6];
	int step_x = 5;
	int scale = 200;
	int offset_y = 30;
	int offset_x = 30;
	point_x *=step_x;

	switch (flag) {
		case 0:
			y = (point_y * scale)/max_err;
			old_y = (old_point_y * scale)/max_err;
			line(screen, (point_x - step_x) + offset_x, (scale + offset_y) - 
						old_y, point_x + offset_x, (scale + offset_y) - y, 9);
			circlefill(screen, point_x + offset_x, 
						(scale + offset_y) - y, 1, 9);

			if ((point_x/step_x)%20 == 0 && (point_x/step_x) > 20) {
				sprintf(label,"%1.3f",point_y);
				textout_ex(screen, font, label, point_x, 
										(scale + offset_y) - y - 30, 9, -1);
			}
			break;
		case 1:
			y = (point_y * scale)/max_err;
			old_y = (old_point_y * scale)/max_err;
			line(screen, (point_x - step_x) + offset_x, 
							(scale + offset_y) - old_y, 
							point_x + offset_x, (scale + offset_y) - y, 12);
			circlefill(screen, point_x + offset_x, 
							(scale + offset_y) - y, 1, 12);

			if ((point_x/step_x)%20 == 0 && (point_x/step_x) > 20) {
				sprintf(label,"%1.3f",point_y);
				textout_ex(screen, font, label, point_x, 
										(scale + offset_y) - y - 50, 12, -1);
			}
			break;
		case 2:
			y = point_y * scale;
			old_y = old_point_y * scale;
			line(screen, (point_x - step_x) + offset_x, 
							(scale + offset_y)*2 - old_y, point_x + offset_x, 
							(scale + offset_y)*2 - y, 9);
			circlefill(screen, point_x + offset_x, 
							(scale + offset_y)*2 - y, 1, 9);

			if ((point_x/step_x)%20 == 0 && (point_x/step_x) > 20) {
				sprintf(label,"%1.3f",point_y);
				textout_ex(screen, font, label, point_x, 
							(scale + offset_y)*2 - y + 30, 9, -1);
			}
			break;
		case 3:
			y = point_y * scale;
			old_y = old_point_y * scale;
			line(screen, (point_x - step_x) + offset_x, 
							(scale + offset_y)*2 - old_y, 
							point_x + offset_x, (scale + offset_y)*2 - y, 12);
			circlefill(screen, point_x + offset_x, 
							(scale + offset_y)*2 - y, 1, 12);

			if ((point_x/step_x)%20 == 0 && (point_x/step_x) > 20) {
				sprintf(label,"%1.3f",point_y);
				textout_ex(screen, font, label, point_x, 
									(scale + offset_y)*2 - y + 50, 12, -1);
			}
			break;
		default:
			return;
	}
	return;
}

/**
* @brief Init allegro display
*/
void init_display() {

	int x,y;

	allegro_init();
	install_keyboard();

	set_color_depth(8);
	set_gfx_mode(GFX_AUTODETECT_WINDOWED,DISPLAY_WIDTH,DISPLAY_HEIGHT,0,0);
	clear_to_color(screen,15);

	textout_ex(screen,font,"E_TS: ",120,30,9,15);
	textout_ex(screen,font,"E_VS: ",120,50,12,15);

	textout_ex(screen,font,"A_TS: ",120,400,9,15);
	textout_ex(screen,font,"A_VS: ",120,420,12,15);
	textout_ex(screen,font,"MAX_VS: ",120,440,12,15);

	textout_ex(screen,font,"Epoches: ",800,50,0,15);
	

	for (x = 20; x < 1030; ++x) {
		putpixel(screen,x,230,0);
		putpixel(screen,x,460,0);
	}

	for (y = 20; y < 240; ++y) {
		putpixel(screen,30,y,0);
		putpixel(screen,30,y + 230,0);
	}

	return;
}

/**
* @brief Draw the image of the character if needed
*/
void draw_image (type_data* data) {

	int scale = 12;
	int color;

	for (int k = 0; k < 28; ++k) {
		for (int i = 0; i < 28; ++i) {
			if (data->pixel[i+k*28] == 1) 
				color = makecol(255, 0, 0);
			else
				color = makecol(0, 0, 0);

			for (int h = 0; h < scale; ++h) {
				for (int j = 0; j < scale; ++j) 
					putpixel(screen,i + 20 + (j + i*scale),
											k + 230 + (h + k*scale),color);
			}
		}
	}
	
	return;
}

/**
* @brief Main core
*
* Mandatory Input:
*	1		- # of hidden layer
*	1+n		- # neuron per hidden layer
*	2+n		- # of epochs
*	3+n		- size of mini-batches
*	4+n		- learning rate
*/
int main(int argc, char const *argv[])
{

	/**< Network parameters */
	int layer_number = 2;
	int hidden_layer;
	int *layer_size;
	double learning_rate;
	int max_epoches;
	int epoches_counter = 1;
	int batch_size;
	double epsilon;
	double momentum;

	/**< Display variable */
	char tra_error_label[6] = "";
	char val_error_label[6] = "";

	char tra_acc_label[6] = "";
	char val_acc_label[6] = "";
	char max_val_label[6] = "";

	char epoche_label[5] = "";

	int data_label[63];
	char label_screen[20];

	/**< Softmax variable */
	double net_output;
	double max_outuput;
	double sum_output;

	/**< If 1 is the output layer */
	int output_flag = 0;

	/**< Result variable */
	int max_out_index;
	double max_out_test;

	/**< Sinapsi and layer matricies */
	type_sinapsi **net_sinapsi;
	type_layer **net_layer;

	int i, j, k, r;
	int read_response = 0;

	/**< File descriptors */
	FILE *fp_label;
	FILE *fp_pixel;
	FILE *fp_map;

	char filename[50] = "";

	/**< Labels variable */
	int desired_output[OUTPUT_SIZE];

	/**< Data sets */
	type_data *train_set_used[TRAIN_LOOP];
	type_data *validation_set[VALIDATION_LOOP];
	type_data *test_set_used[TEST_NUMBER];

	/**< Accuracy variable */
	int succ_per_class[OUTPUT_SIZE];
	int size_per_class[OUTPUT_SIZE];
	double accuracy[3];
	double old_accuracy[2];
	int success = 0;
	int check;
	int count;

	/**< Error variable */
	double global_error[2];
	double old_error[2];

	srand(time(NULL));

	strcat(filename, EXAMPLE_TYPE);
	strcat(filename, "_");

	/**<  check if the right number of parameter are given */
	if (argc > 1) {
		if (atoi(argv[1]) > 0) {

			hidden_layer = atoi(argv[1]);

			strcat(filename, argv[1]);
			strcat(filename, "_");

			layer_number += hidden_layer;

			layer_size = (int *)malloc(layer_number * sizeof(int));
			/**< pixel number of images */
			layer_size[0] = INPUT_SIZE;
			/**< number of output */
			layer_size[hidden_layer + 1] = OUTPUT_SIZE; 
		}
		else {
			printf("%s", USAGE);
			return -1;
		}
	}
	else {
		printf("%s", USAGE);
		return -1;
	}

	if (argc == hidden_layer + 7) {

		for (i = 0; i < hidden_layer; ++i) {
			if (atoi(argv[i + 2]) > 0) {
				/**<  neurons of hidden layer*/
				layer_size[i + 1] = atoi(argv[i + 2]);
				strcat(filename, argv[i + 2]);
			}
			else {
				printf("%s", USAGE);
				return -1;
			}
			if (i != hidden_layer - 1)
				strcat(filename, "_");
			else
				strcat(filename, ".txt");
		}

		if (atoi(argv[hidden_layer + 2]) > 0)
			/**< number of iteration*/
			max_epoches = atoi(argv[hidden_layer + 2]); 
		else {
			printf("%s", USAGE);
			return -1;
		}

		if (atoi(argv[hidden_layer + 3]) > 0)
			/**< size of mini batch to compute the gradient*/
			batch_size = atoi(argv[hidden_layer + 3]);
		else {
			printf("%s", USAGE);
			return -1;
		}

		if (atof(argv[hidden_layer + 4]) > 0)
			/**<  learning rate of training phase*/
			learning_rate = atof(argv[hidden_layer + 4]);
		else {
			printf("%s", USAGE);
			return -1;
		}
		if (atof(argv[hidden_layer + 5]) > 0)
			/**<  learning rate of training phase*/
			epsilon = atof(argv[hidden_layer + 5]); 
		else {
			printf("%s", USAGE);
			return -1;
		}
		if (atof(argv[hidden_layer + 6]) > 0)
			/**<  learning rate of training phase*/
			momentum = atof(argv[hidden_layer + 6]); 
		else {
			printf("%s", USAGE);
			return -1;
		}
	}
	else {
		printf("%s", USAGE);
		return -1;
	}

	/*	################ INITIZIALIZING ################ */

	/**<   array with pointer to layers*/
	net_layer = (type_layer **)malloc(layer_number * sizeof(type_layer *));

	/**<   array with pointers to sinapsi*/
	net_sinapsi = (type_sinapsi **)
						malloc((layer_number - 1) * sizeof(type_sinapsi *));

	/**<   initialize the network*/
	for (i = 0; i < layer_number; ++i) {

		net_layer[i] = init_layer(layer_size[i], i);

		/**<   don't create sinapsi if it is the last layer*/
		if (i != layer_number - 1)
			net_sinapsi[i] = init_sinapsi(layer_size[i], layer_size[i + 1]);
	}

	// init display
	init_display();

	/*	################ TRAINING PHASE ################ */

	fp_label = fopen(TRAIN_SET_LABEL, "rb");

	if (!fp_label) {
		printf("Error: file open failed on Label TRAIN.\n");
		perror("Failed: ");
		return -1;
	}

	fp_pixel = fopen(TRAIN_SET_IMAGE, "rb");

	if (!fp_pixel) {
		printf("Error: file open failed on Pixel TRAIN.\n");
		perror("Failed: ");
		return -1;
	}

	/**<   move the file descriptor to the first data index*/
	fseek(fp_label, 4 * 2, SEEK_SET);
	fseek(fp_pixel, 4 * 4, SEEK_SET);

	for (i = 0; i < OUTPUT_SIZE; ++i)
		for (j = 0; j < TOT_PER_CLASS; ++j)
			train_set.classes[i][j] = (type_data *)malloc(sizeof(type_data));
	train_set.sizes[i] = 0;

	for (i = 0; i < TRAIN_LOOP; ++i)
		train_set_used[i] = (type_data *)malloc(sizeof(type_data));

	for (i = 0; i < VALIDATION_LOOP; ++i)
		validation_set[i] = (type_data *)malloc(sizeof(type_data));

	read_response = read_train_set(fp_label, fp_pixel);

	if (read_response < 0) {
		printf("Error: file read failed.\n");
		perror("Failed: ");
		return -1;
	}

	/**< training phase*/
	do
	{
		global_error[0] = 0;

		success = 0;

		/**< shuffling the data sent for the stochastic property*/
		pick_example(train_set_used, validation_set);

		shuffle(train_set_used, TRAIN_LOOP);

		/**< for each image*/
		for (j = 0; j < TRAIN_LOOP; ++j)
		{

			/**< initialize value of input layer*/
			for (k = 0; k < INPUT_SIZE; ++k) {
				net_layer[0]->activation_value[k] = train_set_used[j]->pixel[k];
				net_layer[0]->x_value[k] = train_set_used[j]->pixel[k];
			}

			output_flag = 0;

			/**< propagate result*/
			for (k = 1; k < layer_number; ++k)
			{
				if (k == layer_number - 1)
					output_flag = 1;
				propagate_into_layer(net_layer[k - 1], net_layer[k], 
									net_sinapsi[k - 1], output_flag);
			}

			/**< initialize value of desired output to zero and fill it*/
			memset(desired_output, 0, sizeof(int) * OUTPUT_SIZE);
			desired_output[train_set_used[j]->label] = 1;

			sum_output = 0;
			max_outuput = 0;

			/**< Compute max output probability*/
			for (k = 0; k < OUTPUT_SIZE; ++k) {
				if (net_layer[layer_number - 1]->activation_value[k] > 
																max_outuput)
					max_outuput = net_layer[layer_number - 1]
													->activation_value[k];
			}

			/**< Compute the sum of the exp of each output probability*/
			for (k = 0; k < OUTPUT_SIZE; ++k) {
				net_output = net_layer[layer_number - 1]->activation_value[k];
				sum_output += exp(net_output + max_outuput);
			}

			max_out_test = 0;

			/**<  compute softmax*/
			for (k = 0; k < OUTPUT_SIZE; ++k) {
				net_layer[layer_number - 1]->x_value[k] = 
							softmax(net_layer[layer_number - 1]->
							activation_value[k], sum_output, max_outuput);

				if (max_out_test < net_layer[layer_number - 1]->x_value[k]) {
					max_out_test = net_layer[layer_number - 1]->x_value[k];
					max_out_index = k;
				}
			}

			net_output = net_layer[layer_number - 1]->
											x_value[train_set_used[j]->label];

			/**< Global error */
			global_error[0] -= (desired_output[train_set_used[j]->label] *
								log(net_output));

			if (train_set_used[j]->label == max_out_index)
				success++;

			/**<  compute output delta, weights and bias gradient*/
			compute_output_delta(desired_output, net_layer[layer_number - 1],
								net_sinapsi[layer_number - 2], 
								net_layer[layer_number - 2], learning_rate);

			/**<  backward propagation*/
			for (k = layer_number - 2; k > 0; --k)
				/**<  compute hidden layer delta, weights and bias gradient*/
				compute_layer_delta(net_sinapsi[k - 1], net_sinapsi[k], 
										net_layer[k - 1], net_layer[k], 
										net_layer[k + 1], learning_rate);

			/**<  each mini batch*/
			if (j % batch_size == 0 && j != 0) {
				/**<  update weights and bias and initialize gradients*/
				for (k = 0; k < layer_number - 1; ++k) {
					update_sinapsi(net_sinapsi[k], batch_size, momentum);

					for (r = 0; r < net_sinapsi[k]->card_out; ++r)
						memset(net_sinapsi[k]->gradient_weights[r], 0, 
									sizeof(double) * net_sinapsi[k]->card_in);
					memset(net_sinapsi[k]->gradient_bias, 0, 
									sizeof(double) * net_sinapsi[k]->card_out);
				}
			}
		}

		/**< Compute global error and accuracy*/
		accuracy[0] = ((double)success / TRAIN_LOOP);
		global_error[0] = ((double)global_error[0] / TRAIN_LOOP);

		success = 0;

		global_error[1] = 0;

		/**< Validation phase*/
		for (j = 0; j < VALIDATION_LOOP; ++j) {

			/**< initialize value of input layer*/
			for (k = 0; k < INPUT_SIZE; ++k) {
				net_layer[0]->activation_value[k] = validation_set[j]->pixel[k];
				net_layer[0]->x_value[k] = validation_set[j]->pixel[k];
			}

			output_flag = 0;

			/**< propagate result*/
			for (k = 1; k < layer_number; ++k) {
				if (k == layer_number - 1)
					output_flag = 1;
				propagate_into_layer(net_layer[k - 1], net_layer[k], 
										net_sinapsi[k - 1], output_flag);
			}

			/**< initialize value of desired output to zero and fill it*/
			memset(desired_output, 0, sizeof(int) * OUTPUT_SIZE);
			desired_output[validation_set[j]->label] = 1;

			sum_output = 0;
			max_outuput = 0;
			/**< Compute max output probability*/
			for (k = 0; k < OUTPUT_SIZE; ++k) {
				if (net_layer[layer_number - 1]->activation_value[k] >
															 max_outuput)
					max_outuput = net_layer[layer_number - 1]->
														activation_value[k];
			}
			/**< Compute the sum of the exp of each output probability*/
			for (k = 0; k < OUTPUT_SIZE; ++k) {
				net_output = net_layer[layer_number - 1]->activation_value[k];
				sum_output += exp(net_output + max_outuput);
			}

			max_out_test = 0;

			/**<  compute softmax*/			
			for (k = 0; k < OUTPUT_SIZE; ++k) {
				net_layer[layer_number - 1]->x_value[k] = 
									softmax(net_layer[layer_number - 1]->
									activation_value[k], 
									sum_output, max_outuput);

				if (max_out_test < net_layer[layer_number - 1]->x_value[k]) {
					max_out_test = net_layer[layer_number - 1]->x_value[k];
					max_out_index = k;
				}
			}

			net_output = net_layer[layer_number - 1]->x_value[validation_set[j]->label];

			global_error[1] -= (desired_output[validation_set[j]->label] *
								log(net_output));

			if (validation_set[j]->label == max_out_index)
				success++;
		}

		/**< Compute accuracy and error on validation set*/
		old_accuracy[1] = accuracy[1];
		accuracy[1] = ((double)success / VALIDATION_LOOP);
		global_error[1] = ((double)global_error[1] / VALIDATION_LOOP);

		if (accuracy[1] > val_max_acc)
			val_max_acc = accuracy[1];

		/**< ALLEGRO UTILS*/
		if (epoches_counter == 1) {
			old_error[0] = global_error[0];
			old_error[1] = global_error[1];

			if (global_error[0] > global_error[1])
				max_err = global_error[0];
			else
				max_err = global_error[1];

			old_accuracy[0] = 0;
			old_accuracy[1] = 0;
		}

		draw_point(global_error[0], epoches_counter, old_error[0], 0);
		draw_point(global_error[1], epoches_counter, old_error[1], 1);

		draw_point(accuracy[0], epoches_counter, old_accuracy[0], 2);
		draw_point(accuracy[1], epoches_counter, old_accuracy[1], 3);

		sprintf(tra_error_label, "%2.3f", global_error[0]);
		sprintf(val_error_label, "%2.3f", global_error[1]);

		sprintf(tra_acc_label, "%2.3f", accuracy[0]);
		sprintf(val_acc_label, "%2.3f", accuracy[1]);

		sprintf(max_val_label, "%2.3f", val_max_acc);

		sprintf(epoche_label, "%d", epoches_counter);

		tra_error_label[5] = '\0';
		val_error_label[5] = '\0';
		tra_acc_label[5] = '\0';
		val_acc_label[5] = '\0';
		max_val_label[5] = '\0';

		textout_ex(screen, font, epoche_label, 880, 50, 0, 15);

		textout_ex(screen, font, tra_error_label, 180, 30, 9, 15);
		textout_ex(screen, font, val_error_label, 180, 50, 12, 15);

		textout_ex(screen, font, tra_acc_label, 180, 400, 9, 15);
		textout_ex(screen, font, val_acc_label, 180, 420, 12, 15);
		textout_ex(screen, font, max_val_label, 180, 440, 12, 15);

		old_error[0] = global_error[0];
		old_error[1] = global_error[1];

		old_accuracy[0] = accuracy[0];
		old_accuracy[1] = accuracy[1];

		epoches_counter++;
	} while (epoches_counter <= max_epoches && global_error[0] > epsilon &&
				accuracy[1] > (val_max_acc - val_offset));

	fclose(fp_label);
	fclose(fp_pixel);

	/*	################ TESTING PHASE ################ */

	fp_label = fopen(TEST_SET_LABEL, "rb");

	if (!fp_label)
	{
		printf("Error: file open failed on Label TEST.\n");
		return -1;
	}

	fp_pixel = fopen(TEST_SET_IMAGE, "rb");

	if (!fp_pixel)
	{
		printf("Error: file open failed on Pixel TEST.\n");
		perror("Failed: ");
		return -1;
	}

	/**< move the file descriptor to the first data index*/
	fseek(fp_label, 4 * 2, SEEK_SET);
	fseek(fp_pixel, 4 * 4, SEEK_SET);

	for (i = 0; i < TEST_NUMBER; ++i)
		test_set_used[i] = (type_data *)malloc(sizeof(type_data));

	read_response = read_test_set(fp_label, fp_pixel, test_set_used);

	if (read_response < 0)
	{
		printf("Error: file read failed.\n");
		perror("Failed: ");
		return -1;
	}

	memset(size_per_class, 0, OUTPUT_SIZE * sizeof(int));
	memset(succ_per_class, 0, OUTPUT_SIZE * sizeof(int));

	success = 0;

	/**< for each image*/
	for (j = 0; j < TEST_NUMBER; ++j) {

		/**< initialize value of input layer*/
		for (k = 0; k < INPUT_SIZE; ++k) {
			net_layer[0]->activation_value[k] = test_set_used[j]->pixel[k];
			net_layer[0]->x_value[k] = test_set_used[j]->pixel[k];
		}

		output_flag = 0;

		/**< propagate result*/
		for (k = 1; k < layer_number; ++k) {
			if (k == layer_number - 1)
				output_flag = 1;
			propagate_into_layer(net_layer[k - 1], net_layer[k], 
										net_sinapsi[k - 1], output_flag);
		}

		max_outuput = 0;
		sum_output = 0;

		/**< Comput the highest ouput value*/
		for (k = 0; k < OUTPUT_SIZE; ++k) {
			if (net_layer[layer_number - 1]->activation_value[k] > max_outuput)
				max_outuput = net_layer[layer_number - 1]->activation_value[k];
		}
		/**< Comput the sum of the exp of all ouput probability*/
		for (k = 0; k < OUTPUT_SIZE; ++k) {
			net_output = net_layer[layer_number - 1]->activation_value[k];
			sum_output += exp(net_output + max_outuput);
		}

		max_out_test = 0;

		/**< Compute the highest probability value*/
		for (k = 0; k < OUTPUT_SIZE; ++k) {
			net_layer[layer_number - 1]->x_value[k] = 
							softmax(net_layer[layer_number - 1]->
							activation_value[k], sum_output, max_outuput);

			if (max_out_test < net_layer[layer_number - 1]->x_value[k]) {
				max_out_test = net_layer[layer_number - 1]->x_value[k];
				max_out_index = k;
			}
		}
		/**< Check the result */
		if (test_set_used[j]->label == max_out_index) {
			success++;
			succ_per_class[test_set_used[j]->label]++;
		}

		size_per_class[test_set_used[j]->label]++;
	}


	fclose(fp_label);
	fclose(fp_pixel);

	accuracy[2] = (double)((success * 100) / TEST_NUMBER);

	printf("Success: %d\nTotal: %d\n", success, TEST_NUMBER);
	printf("%2.2f%% of accuracy!\n", accuracy[2]);

	for (i = 0; i < OUTPUT_SIZE; ++i) {
		if (size_per_class[i] != 0)
			printf("Label: %d -> %2.2f%% Accurancy!\n", i, 
				(double)((succ_per_class[i] * 100) / size_per_class[i]));
	}

	save_file(net_sinapsi, layer_number - 1, accuracy[2], filename);

	/*	################ FREE PHASE ################ */

	for (int i = 0; i < layer_number; ++i)
	{
		free(net_layer[i]->activation_value);
		free(net_layer[i]->x_value);
		if (i > 0)
			free(net_layer[i]->delta);

		if (i != layer_number - 1)
		{
			for (j = 0; j < net_sinapsi[i]->card_out; ++j)
				free(net_sinapsi[i]->weights[j]);

			free(net_sinapsi[i]->weights);

			free(net_sinapsi[i]->bias);

			for (j = 0; j < net_sinapsi[i]->card_out; ++j)
				free(net_sinapsi[i]->gradient_weights[j]);

			free(net_sinapsi[i]->gradient_weights);

			free(net_sinapsi[i]->gradient_bias);

			free(net_sinapsi[i]);
		}

		free(net_layer[i]);
	}

	free(layer_size);
	free(net_layer);
	free(net_sinapsi);

	readkey();

	allegro_exit();

	return 0;
}

