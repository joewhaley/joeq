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

    static final int CACHE_LINE_SIZE = 256;
    
    static final byte BOUNDS_EX_NUM = 5;
    
    static final int EAX = 0;
    static final int ECX = 1;
    static final int EDX = 2;
    static final int EBX = 3;
    static final int ESP = 4;
    static final int EBP = 5;
    static final int ESI = 6;
    static final int EDI = 7;

    static final int AL = 0;

    static final int AX = 0;

    static final int RA = 0x04;
    static final int SEIMM8 = 0x0200;
    static final int SHIFT_ONCE = 0x1000;
    static final int CJUMP_SHORT = 0x70;
    static final int CJUMP_NEAR = 0x0F80;
    static final int JUMP_SHORT = 0x0B;
    static final int JUMP_NEAR = 0x09;
    
    static final int MOD_EA = 0x00;
    static final int MOD_DISP8 = 0x40;
    static final int MOD_DISP32 = 0x80;
    static final int MOD_REG = 0xC0;

    static final int RM_SIB = 0x04;

    static final int SCALE_1 = 0x00;
    static final int SCALE_2 = 0x40;
    static final int SCALE_4 = 0x80;
    static final int SCALE_8 = 0xC0;

    // pairing
    static final int NP = 0;
    static final int PU = 1;
    static final int PV = 2;
    static final int UV = 3;

    // u-ops
    static final int COMPLEX = 5;
}
