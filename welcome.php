<?php 
header("Content-Type:text/html; charset=big5");
session_start();

if (!isset($_SESSION['username'])) {
    header("Location: index.php");
}

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>�ݨ�����</title>
</head>
<body>
    <?php echo "<h1 >��ꭷ�I�լd " . $_SESSION['username'] . "</h1>"; ?>
    <!--
    <div style="text-align:right;"><a href="logout.php">�n�X</a>
    -->
    <div style="text-align:right;"><input type="button" value="�n�X"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='logout.php'"></div>
    <div style="text-align:center;">

                              <form name="���" method="post" action="Ex5_4-action.php">

                                  <h3>�Ǿ�</h3>
                                  
                                  <br>
                                  <!--
                                  �]�wselect ���ȥh�p�������
                                  -->
                                  <select id="sample3">
                                      <option value="1">�п�ܧA���̰��Ǿ�</option>
                                      <option value="2">�ꤤ(�t)�H�U</option>
                                      <option value="3">����¾</option>
                                      <option value="4">�j��</option>
                                      <option value="5">��s��</option>
                                     

                                  </select>

                                  <h3>¾�~���O</h3>
                                  <br>
                                  <select id="sample4">
                                      <option value="1">���īO�I</option>
                                      <option value="1">�F������Ʒ~</option>
                                      <option value="1">�xĵ����</option>
                                      <option value="1">�����Ш|</option>
                                      <option value="1">�\���[���Ǽ��~</option>
                                      <option value="1">�B����x</option>
                                      <option value="1">��T���</option>
                                      <option value="1">�s�y�ؿv</option>
                                      <option value="1">�D��Q���c/�v�Ъk�H</option>
                                      <option value="1">�h����c</option>
                                      <option value="1">���N�N��/�F�v�H��</option>
                                      <option value="1">�a��</option>
                                      <option value="1">�ǥ�</option>
                                      <option value="1">�ݷ~��</option>
                                      <option value="1">�i�X�f�T��</option>
                                      <option value="1">�߮v/���ҤH/�|�p�v</option>
                                      <option value="1">�ȼ�/��Q</option>
                                      <option value="1">�a�F�h�Τ��ʲ��g��</option>
                                      <option value="1">�O�b�h</option>
                                      <option value="1">��L</option>
                                  </select>

                                  <h3>¾��</h3>
                                  <br>
                                  <select id="sample5">
                                      <option value="1">¾��</option>
                                      <option value="2">�~��</option>
                                      <option value="3">�޳N�H��</option>
                                      <option value="4">�����D��</option>
                                      <option value="5">�����D��</option>
                                      <option value="6">���~�t�d�H</option>
                                      <option value="7">��L</option>
                                  </select>

                                  <h3>�a�x�έӤH�~���J</h3>
                                  <br>
                                  <select id="sample6">
                                      <option value="1">50�U�H�U</option>
                                      <option value="2">50�U-100�U</option>
                                      <option value="3">100�U-300�U</option>
                                      <option value="4">300-500�U</option>
                                      <option value="5">500�U�H�W</option>

                                  </select>

                                  <h3>�i�����B</h3>
                                  <br>
                                  <select id="sample7">
                                      <option value="1">50�U�H�U</option>
                                      <option value="2">50�U-100�U</option>
                                      <option value="3">100�U-300�U</option>
                                      <option value="4">300-500�U</option>
                                      <option value="5">500�U-1000�U</option>
                                      <option value="6">1000�U-3000�U</option>
                                      <option value="7">3000�U�H�W</option>
                                  </select>

                                  <h3>�O�_�⦳�������d�O�I���j�˯f�ҩ��H</h3>
                                  <br>
                                  <select id="sample8">
                                      <option value="1">�O</option>
                                      <option value="2">�_</option>

                                  </select>

                                  <h1>���I�A�X�׵�����</h1>
                                  <br>
                                  <h3>�Ȥ�~�ּh</h3>
                                  <select id="sample9">
                                      <option value="1">20-30��</option>
                                      <option value="2">30-40��</option>
                                      <option value="3">40-50��</option>
                                      <option value="4">50���H�W</option>
                                  </select>
                                  <br>
                                  <h3>�z�����g��</h3>
                                  <select id="sample10">
                                      <option value="1">1�~�H�U</option>
                                      <option value="2">1�~-3�~</option>
                                      <option value="3">3�~-5�~</option>
                                      <option value="4">5�~�H�W</option>
                                  </select>
                                  <br>
                                  <h3>�аݱz�����J���h�֤�ҥi�Ω�����x�W�H</h3>
                                  <br>
                                  <select id="sample11">
                                      <option value="1">0%-5%</option>
                                      <option value="2">6%-10%</option>
                                      <option value="3">11%-20%</option>
                                      <option value="4">20%-30%</option>
                                      <option value="5">30%�H�W</option>
                                  </select>
                                  <h3>�аݱz�w�p�h�[��ū�^�z��������H</h3>
                                  <br>
                                  <select id="sample12">
                                      <option value="1">�b�~�H�U</option>
                                      <option value="2">�b�~-1�~</option>
                                      <option value="3">1�~-2�~</option>
                                      <option value="4">3�~�H�W</option>
                                  </select>
                                  <h3>����i�ʪ�������,�z�i�H����������T�סH</h3>
                                  <br>
                                  <select id="sample13">
                                      <option value="1">���t5%</option>
                                      <option value="2">���t10%</option>
                                      <option value="3">���t20%</option>
                                      <option value="4">���t30%</option>
                                  </select>
                                  <h3>�U�C��̳̱���z�����欰�P�g��H</h3>
                                  <br>
                                  <select id="sample2">
                                      <option value="1">�ߦní�w���q</option>
                                      <option value="2">�Ӿ�ֶq���I�A�����O���S</option>
                                      <option value="3">�Ӿ�A���I�A����A����S</option>
                                      <option value="4">�Ӿᰪ���I�A��������S</option>
                                  </select>

                                  <h3>��z�w�g�}�l�l���F�A�o�ɱz�|�H</h3>
                                  <select id="sample">
                                      <option value="1">����ū�^</option>
                                      <option value="2">����ū�^</option>
                                      <option value="3">�~���[��</option>
                                      <option value="4">����[�X</option>
                                  </select>

                                  <h1>�A�������Ƭ�:</h1>
                                  <div id="test"><h2>0</h2></div><h2>��</h2>
                                  <h1>�A�A�X��ꭷ�欰:</h1>
                                  <div id="investresult"><h2></h2></div>
                                  <br>

                                  <!--
                                  ���P�_
                                  20< print("�O����")
                                  20-40 print("���ū�")
                                  >40 print("������")
                                  -->



                                   <script>
                                        function myTest() {
                                            var sampleValue = parseInt(document.getElementById("sample").value);
                                            var sampleValue2 = parseInt(document.getElementById("sample2").value);
                                            var sampleValue3 = parseInt(document.getElementById("sample3").value);
                                            var sampleValue4 = parseInt(document.getElementById("sample4").value);
                                            var sampleValue5 = parseInt(document.getElementById("sample5").value);
                                            var sampleValue6 = parseInt(document.getElementById("sample6").value);
                                            var sampleValue7 = parseInt(document.getElementById("sample7").value);
                                            var sampleValue8 = parseInt(document.getElementById("sample8").value);
                                            var sampleValue9 = parseInt(document.getElementById("sample9").value);
                                            var sampleValue10 = parseInt(document.getElementById("sample10").value);
                                            var sampleValue11 = parseInt(document.getElementById("sample11").value);
                                            var sampleValue12 = parseInt(document.getElementById("sample12").value);
                                            var sampleValue13 = parseInt(document.getElementById("sample13").value);

                                            var result = sampleValue + sampleValue2 +�@sampleValue3�@+sampleValue4
                                            +sampleValue5 +sampleValue6 +sampleValue7 +sampleValue8 +sampleValue9
                                            +sampleValue10 +sampleValue11+sampleValue12+sampleValue13;
                                            

                                            
                                            document.getElementById("test").innerHTML = result;

                                         


                                            if (result < 20) {
	                                              alert('�A�A�X���O�O�������')
                                                  document.getElementById("investresult").innerHTML = '�A�A�X���O�O�������';
                                             } else if (20 < result < 40) {
                                                  alert('�A�A�X���O���ū����')
                                                  document.getElementById("investresult").innerHTML = '�A�A�X���O���ū����';
                                             } else {
                                                  alert('�A�A�X���O���������')
                                                  document.getElementById("investresult").innerHTML = '�A�A�X���O���������';
                                             }


                                          }

                                           
                                           

                                                
                                    </script>





                              </form>

                              

                               

                               <div style="text-align:center;"><input type="submit" value="�p��"  style="width:120px;height:40px;font-size:20px;" onclick="myTest() "></div>
                              <br>
                                
                              <br>
                              
                               

                               <form action="Selection.php" method="post">
                               


                                        

                                 <input type="submit" value="�U�@�B"  
                               style="width:120px;height:40px;font-size:20px;" onclick="location.href='Selection.php'">
                              

                               <div style="text-align:center;">

                              </form>

                              
                              
                               
                               </div>
                          </div>

</body>
</html>