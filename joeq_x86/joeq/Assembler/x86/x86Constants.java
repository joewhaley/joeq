/*
 * x86Code.java
 *
 * Created on December 22, 2000, 7:13 AM
 *
 */

package Assembler.x86;

/*
 * @author  John Whaley
 * @version $Id$
 */
public interface x86Constants {

    int CACHE_LINE_SIZE = 256;
    
    byte BOUNDS_EX_NUM = 5;
    
    int EAX = 0;
    int ECX = 1;
    int EDX = 2;
    int EBX = 3;
    int ESP = 4;
    int EBP = 5;
    int ESI = 6;
    int EDI = 7;

    int AL = 0;

    int AX = 0;

    int RA = 0x04;
    int SEIMM8 = 0x0200;
    int SHIFT_ONCE = 0x1000;
    int CJUMP_SHORT = 0x70;
    int CJUMP_NEAR = 0x0F80;
    int JUMP_SHORT = 0x0B;
    int JUMP_NEAR = 0x09;
    
    int MOD_EA = 0x00;
    int MOD_DISP8 = 0x40;
    int MOD_DISP32 = 0x80;
    int MOD_REG = 0xC0;

    int RM_SIB = 0x04;

    int SCALE_1 = 0x00;
    int SCALE_2 = 0x40;
    int SCALE_4 = 0x80;
    int SCALE_8 = 0xC0;

    // pairing
    int NP = 0;
    int PU = 1;
    int PV = 2;
    int UV = 3;

    // u-ops
    int COMPLEX = 5;
}
