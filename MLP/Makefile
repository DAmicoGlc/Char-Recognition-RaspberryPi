CC = gcc

OBJS = objs

ALLEGRO_FLAG = `allegro-config --libs`

USERLAND_ROOT = $(HOME)/git/raspberrypi/userland
CFLAGS_PI = \
	-I$(USERLAND_ROOT)/host_applications/linux/libs/bcm_host/include \
	-I$(USERLAND_ROOT)/host_applications/linux/apps/raspicam \
	-I$(USERLAND_ROOT) \
	-I$(USERLAND_ROOT)/interface/vcos/pthreads \
	-I$(USERLAND_ROOT)/interface/vmcs_host/linux \
	-I$(USERLAND_ROOT)/interface/mmal

LDFLAGS_PI = -L$(USERLAND_ROOT)/build/lib -lmmal_core -lmmal -l mmal_util -lvcos -lbcm_host

CFLAGS = -Wno-multichar -g $(CFLAGS_PI) -MD

LDFLAGS = $(LDFLAGS_PI) -lpthread -lm

RASPICAM_OBJS = \
	$(OBJS)/RaspiCamControl.o \
	$(OBJS)/RaspiCLI.o 

PROJECT_OBJS = \
	$(OBJS)/hand_written_recognition.o \
	$(OBJS)/raspi_cam.o \
	$(OBJS)/ptask_handler.o \
	$(OBJS)/user.o \
	$(OBJS)/display.o \
	$(OBJS)/nn_handler.o

TARGETS = hand_written_recognition

all: $(TARGETS)

$(OBJS)/%.o: %.c
	$(CC) -c $(CFLAGS) $(ALLEGRO_FLAG) $< -o $@

$(OBJS)/%.o: $(USERLAND_ROOT)/host_applications/linux/apps/raspicam/%.c
	$(CC) -c $(CFLAGS) $< -o $@

libraspicam.a: $(RASPICAM_OBJS)
	ar rcs libraspicam.a -o $+

hand_written_recognition: $(PROJECT_OBJS) libraspicam.a
	$(CC) $(LDFLAGS) $+ $(ALLEGRO_FLAG) -L. -lraspicam -o $@

clean:
	rm -f $(OBJS)/* $(TARGETS)

-include $(OBJS)/*.d