ARCH=arm64

CPP=/opt/poky/4.0.2/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++
CPPFLAGS  = --sysroot=/opt/poky/4.0.2/sysroots/armv8a-poky-linux
CPPFLAGS += -g 

CPPFLAGS   += -std=c++17 -O2 -D_FORTIFY_SOURCE=2 -pipe -g -MMD -MP -Wall -Wextra -Winit-self -Wno-missing-field-initializers
LDFLAGS   = --sysroot=/opt/poky/4.0.2/sysroots/armv8a-poky-linux
LIBS      = -L/opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib -lhailo_medialibrary_api -lhailo_media_library_common -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

INCLUDE   = -I./include -I/opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/include/hailo -I/opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/include/opencv4
TARGET    = ./bin/$(shell basename `readlink -f .`)
SRCDIR    = ./src
ifeq "$(strip $(SRCDIR))" ""
  SRCDIR  = .
endif
SOURCES   = $(wildcard $(SRCDIR)/*.cpp)
OBJDIR    = ./obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR  = .
endif
OBJECTS   = $(addprefix $(OBJDIR)/$(ARCH)/, $(notdir $(SOURCES:.cpp=.o)))
DEPENDS   = $(OBJECTS:.o=.d)

$(TARGET).$(ARCH): $(OBJECTS)
	mkdir -p $(dir $(TARGET))
	$(CPP) $^ $(LDFLAGS) $(LIBS) -o $@

$(OBJDIR)/$(ARCH)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)/$(ARCH)
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

clean:
	-rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)

