#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

int main(int argc, char** argv)
{
  int fd;
  unsigned int *p, *q;
  unsigned int e_phoff;
  unsigned int start, length;
  char* page_start;
  unsigned short i, e_phnum;
  size_t page_size = getpagesize();

  if (argc < 2) {
    printf("Usage: %s <executable file>\n", argv[0]);
    return 0;
  }

  fd = open(argv[1], O_RDWR);
  if (!fd) {
    printf("Cannot open file %s.\n", argv[1]);
    return 1;
  }

  p = (int*)mmap(0, 52, PROT_READ, MAP_SHARED, fd, 0);
  if ((void*)p == MAP_FAILED) {
    printf("Cannot map file %s offset 0.\n", argv[1]);
    return 1;
  }
  e_phoff = *(p+7);
  e_phnum = *(unsigned short *)(p+11);
  printf("Program header table at offset %d, %d entries.\n", e_phoff, e_phnum);
  munmap((void*)p, 52);

  start = e_phoff / page_size * page_size;
  length = (e_phoff + 32*e_phnum - start + page_size - 1) / page_size * page_size;
  page_start = (char*)mmap(0, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, start);
  if (page_start == MAP_FAILED) {
    printf("Cannot map file %s offset %d size %d.\n", argv[1], start, length);
    return 1;
  }
  q = p = (int*)(page_start + e_phoff - start);
  for (i=0; i<e_phnum; ++i) {
    printf("Program header %d type %x flags %x\n", i, q[0], q[6]);
    if (q[0] == 1) { /* PT_LOAD */
      if (q[6] == 5) { /* PF_R | PF_X */
	printf("Adding writeable flag.\n");
	q[6] = 7;
      }
    }
    q += 8;
  }
  munmap((void*)p, 32*e_phnum);
  close(fd);
  return 0;
}
