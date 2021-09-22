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
    <title>問卷評估</title>
</head>
<body>
    <?php echo "<h1 >投資風險調查 " . $_SESSION['username'] . "</h1>"; ?>
    <!--
    <div style="text-align:right;"><a href="logout.php">登出</a>
    -->
    <div style="text-align:right;"><input type="button" value="登出"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='logout.php'"></div>
    <div style="text-align:center;">

                              <form name="表單" method="post" action="Ex5_4-action.php">

                                  <h3>學歷</h3>
                                  
                                  <br>
                                  <!--
                                  設定select 的值去計算投資分數
                                  -->
                                  <select id="sample3">
                                      <option value="1">請選擇你的最高學歷</option>
                                      <option value="2">國中(含)以下</option>
                                      <option value="3">高中職</option>
                                      <option value="4">大學</option>
                                      <option value="5">研究所</option>
                                     

                                  </select>

                                  <h3>職業類別</h3>
                                  <br>
                                  <select id="sample4">
                                      <option value="1">金融保險</option>
                                      <option value="1">政府公營事業</option>
                                      <option value="1">軍警消防</option>
                                      <option value="1">醫療教育</option>
                                      <option value="1">餐飲觀光傳播業</option>
                                      <option value="1">運輸倉儲</option>
                                      <option value="1">資訊科技</option>
                                      <option value="1">製造建築</option>
                                      <option value="1">非營利機構/宗教法人</option>
                                      <option value="1">退休機構</option>
                                      <option value="1">民意代表/政治人物</option>
                                      <option value="1">家管</option>
                                      <option value="1">學生</option>
                                      <option value="1">待業中</option>
                                      <option value="1">進出口貿易</option>
                                      <option value="1">律師/公證人/會計師</option>
                                      <option value="1">銀樓/當鋪</option>
                                      <option value="1">地政士及不動產經紀</option>
                                      <option value="1">記帳士</option>
                                      <option value="1">其他</option>
                                  </select>

                                  <h3>職務</h3>
                                  <br>
                                  <select id="sample5">
                                      <option value="1">職員</option>
                                      <option value="2">業務</option>
                                      <option value="3">技術人員</option>
                                      <option value="4">中階主管</option>
                                      <option value="5">高階主管</option>
                                      <option value="6">企業負責人</option>
                                      <option value="7">其他</option>
                                  </select>

                                  <h3>家庭或個人年收入</h3>
                                  <br>
                                  <select id="sample6">
                                      <option value="1">50萬以下</option>
                                      <option value="2">50萬-100萬</option>
                                      <option value="3">100萬-300萬</option>
                                      <option value="4">300-500萬</option>
                                      <option value="5">500萬以上</option>

                                  </select>

                                  <h3>可投資金額</h3>
                                  <br>
                                  <select id="sample7">
                                      <option value="1">50萬以下</option>
                                      <option value="2">50萬-100萬</option>
                                      <option value="3">100萬-300萬</option>
                                      <option value="4">300-500萬</option>
                                      <option value="5">500萬-1000萬</option>
                                      <option value="6">1000萬-3000萬</option>
                                      <option value="7">3000萬以上</option>
                                  </select>

                                  <h3>是否領有全民健康保險重大傷病證明？</h3>
                                  <br>
                                  <select id="sample8">
                                      <option value="1">是</option>
                                      <option value="2">否</option>

                                  </select>

                                  <h1>風險適合度評分表</h1>
                                  <br>
                                  <h3>客戶年齡層</h3>
                                  <select id="sample9">
                                      <option value="1">20-30歲</option>
                                      <option value="2">30-40歲</option>
                                      <option value="3">40-50歲</option>
                                      <option value="4">50歲以上</option>
                                  </select>
                                  <br>
                                  <h3>您的投資經驗</h3>
                                  <select id="sample10">
                                      <option value="1">1年以下</option>
                                      <option value="2">1年-3年</option>
                                      <option value="3">3年-5年</option>
                                      <option value="4">5年以上</option>
                                  </select>
                                  <br>
                                  <h3>請問您的收入有多少比例可用於投資或儲蓄？</h3>
                                  <br>
                                  <select id="sample11">
                                      <option value="1">0%-5%</option>
                                      <option value="2">6%-10%</option>
                                      <option value="3">11%-20%</option>
                                      <option value="4">20%-30%</option>
                                      <option value="5">30%以上</option>
                                  </select>
                                  <h3>請問您預計多久後贖回您的基金投資？</h3>
                                  <br>
                                  <select id="sample12">
                                      <option value="1">半年以下</option>
                                      <option value="2">半年-1年</option>
                                      <option value="3">1年-2年</option>
                                      <option value="4">3年以上</option>
                                  </select>
                                  <h3>價格波動的市場中,您可以接受的價格幅度？</h3>
                                  <br>
                                  <select id="sample13">
                                      <option value="1">正負5%</option>
                                      <option value="2">正負10%</option>
                                      <option value="3">正負20%</option>
                                      <option value="4">正負30%</option>
                                  </select>
                                  <h3>下列何者最接近您的投資行為與經驗？</h3>
                                  <br>
                                  <select id="sample2">
                                      <option value="1">喜好穩定收益</option>
                                      <option value="2">承擔少量風險，獲取潛力報酬</option>
                                      <option value="3">承擔適當風險，獲取適當報酬</option>
                                      <option value="4">承擔高風險，獲取高報酬</option>
                                  </select>

                                  <h3>當您已經開始損失了，這時您會？</h3>
                                  <select id="sample">
                                      <option value="1">全部贖回</option>
                                      <option value="2">部分贖回</option>
                                      <option value="3">繼續觀察</option>
                                      <option value="4">持續加碼</option>
                                  </select>

                                  <h1>你的投資分數為:</h1>
                                  <div id="test"><h2>0</h2></div><h2>分</h2>
                                  <h1>你適合投資風格為:</h1>
                                  <div id="investresult"><h2></h2></div>
                                  <br>

                                  <!--
                                  做判斷
                                  20< print("保本型")
                                  20-40 print("平衡型")
                                  >40 print("成長型")
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

                                            var result = sampleValue + sampleValue2 +　sampleValue3　+sampleValue4
                                            +sampleValue5 +sampleValue6 +sampleValue7 +sampleValue8 +sampleValue9
                                            +sampleValue10 +sampleValue11+sampleValue12+sampleValue13;
                                            

                                            
                                            document.getElementById("test").innerHTML = result;

                                         


                                            if (result < 20) {
	                                              alert('你適合的是保本型投資')
                                                  document.getElementById("investresult").innerHTML = '你適合的是保本型投資';
                                             } else if (20 < result < 40) {
                                                  alert('你適合的是平衡型投資')
                                                  document.getElementById("investresult").innerHTML = '你適合的是平衡型投資';
                                             } else {
                                                  alert('你適合的是成長型投資')
                                                  document.getElementById("investresult").innerHTML = '你適合的是成長型投資';
                                             }


                                          }

                                           
                                           

                                                
                                    </script>





                              </form>

                              

                               

                               <div style="text-align:center;"><input type="submit" value="計算"  style="width:120px;height:40px;font-size:20px;" onclick="myTest() "></div>
                              <br>
                                
                              <br>
                              
                               

                               <form action="Selection.php" method="post">
                               


                                        

                                 <input type="submit" value="下一步"  
                               style="width:120px;height:40px;font-size:20px;" onclick="location.href='Selection.php'">
                              

                               <div style="text-align:center;">

                              </form>

                              
                              
                               
                               </div>
                          </div>

</body>
</html>